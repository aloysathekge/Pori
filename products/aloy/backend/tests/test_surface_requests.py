from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlmodel import select

from aloy_backend import background as background_module
from aloy_backend.model_roles import ModelAssignment, ModelRole
from aloy_backend.models import (
    Event,
    EventTrailEntry,
    Organization,
    OrganizationMembership,
    Run,
    SurfaceBuild,
    SurfaceProject,
    SurfacePublication,
    SurfaceRevision,
)
from aloy_backend.run_profiles import SURFACE_BUILDER_RUN_PROFILE
from aloy_backend.runtime import authenticated_run_context
from aloy_backend.surface_requests import (
    SURFACE_BUILDER_RUN_KIND,
    SurfaceBuilderCompletionGuard,
    SurfaceRequestHandler,
    SurfaceRequestParams,
    verified_surface_publication,
)
from aloy_backend.tools.surface_builds import SURFACE_BUILD_CONTEXT_KEY
from aloy_backend.tools.surface_completion import SURFACE_COMPLETION_CONTEXT_KEY
from aloy_backend.tools.surfaces import SURFACE_AUTHORING_CONTEXT_KEY
from pori import stable_fingerprint


def _builder_assignment() -> ModelAssignment:
    values = {
        "config_version": 1,
        "role": ModelRole.SURFACE_BUILDER,
        "provider": "openai",
        "model": "frontier-builder",
        "temperature": 0.1,
        "max_tokens": 16000,
        "reasoning_mode": "none",
        "capabilities": ("structured_output", "tools", "vision"),
        "skill_id": "surface-builder@1",
        "qualification_status": "qualified",
        "qualification_suite": "aloy-surface-builder-v1",
        "qualification_evidence": "eval:builder-2026-07",
    }
    fingerprint = stable_fingerprint(
        {
            **values,
            "role": ModelRole.SURFACE_BUILDER.value,
            "capabilities": list(values["capabilities"]),
        }
    )
    return ModelAssignment.model_validate(
        {
            **values,
            "config_fingerprint": fingerprint,
            "resolution_ms": 1.25,
            "resolved_at": datetime.now(timezone.utc),
        }
    )


async def _create_event(client) -> dict:
    response = await client.post(
        "/v1/events",
        json={
            "title": "University 2026",
            "summary": "Manage courses, timetable, and assessments",
            "phase": "semester",
        },
    )
    assert response.status_code == 201
    return response.json()


def _run_context(event_id: str):
    return authenticated_run_context(
        user_id="test-user",
        organization_id="user:test-user",
        run_id="conversation-run-1",
        session_id="conversation-1",
        event_id=event_id,
        workspace_id=event_id,
        agent_id="default-agent",
        max_steps=15,
    )


async def test_model_surface_request_queues_one_purpose_scoped_builder(
    client,
    db_session_maker,
):
    event = await _create_event(client)
    assignment = _builder_assignment()
    handler = SurfaceRequestHandler(
        run_context=_run_context(event["id"]),
        session_factory=db_session_maker,
        model_assignment_resolver=lambda *_args, **_kwargs: assignment,
    )
    params = SurfaceRequestParams(
        goal="Help me understand and use my semester timetable",
        experience="A weekly timetable with course and assessment views",
        jobs=["See this week", "Open a course", "Plan exam preparation"],
        source_refs=["university_timetable_2026.md"],
    )

    queued = await handler.request(params)
    replayed = await handler.request(params)

    assert queued["status"] == "queued"
    assert queued["ready"] is False
    assert replayed["run_id"] == queued["run_id"]
    assert replayed["replayed"] is True
    async with db_session_maker() as session:
        runs = list(
            (await session.execute(select(Run).where(Run.event_id == event["id"])))
            .scalars()
            .all()
        )
        builder_runs = [run for run in runs if run.run_kind == SURFACE_BUILDER_RUN_KIND]
        assert len(builder_runs) == 1
        run = builder_runs[0]
        assert run.run_profile == SURFACE_BUILDER_RUN_PROFILE.descriptor()
        assert run.model_assignment == assignment.descriptor()
        assert run.agent_id == "surface-builder"
        assert run.parent_run_id == "conversation-run-1"
        assert "Never convert schedule rows" in run.task
        trails = list(
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.event_id == event["id"],
                        EventTrailEntry.kind == "surface_build_queued",
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(trails) == 1


async def test_surface_ready_receipt_requires_this_runs_live_publication(
    client,
    db_session_maker,
):
    event = await _create_event(client)
    assignment = _builder_assignment()
    handler = SurfaceRequestHandler(
        run_context=_run_context(event["id"]),
        session_factory=db_session_maker,
        model_assignment_resolver=lambda *_args, **_kwargs: assignment,
    )
    queued = await handler.request(
        SurfaceRequestParams(
            goal="Show my timetable",
            experience="A useful weekly timetable",
        )
    )

    async with db_session_maker() as session:
        run = await session.get(Run, queued["run_id"])
        assert run is not None
        project = SurfaceProject(
            id="surface-university",
            organization_id=run.organization_id,
            user_id=run.user_id,
            event_id=run.event_id,
        )
        revision = SurfaceRevision(
            id="revision-university",
            organization_id=run.organization_id,
            user_id=run.user_id,
            event_id=run.event_id,
            project_id=project.id,
            revision_number=1,
            creator_run_id=run.id,
            idempotency_key="revision-idempotency",
            request_fingerprint="revision-fingerprint",
            checksum="revision-checksum",
        )
        build = SurfaceBuild(
            id="build-university",
            organization_id=run.organization_id,
            user_id=run.user_id,
            event_id=run.event_id,
            project_id=project.id,
            revision_id=revision.id,
            creator_run_id=run.id,
            idempotency_key="build-idempotency",
            request_fingerprint="build-fingerprint",
            status="succeeded",
            source_checksum=revision.checksum,
            toolchain_version="test",
        )
        publication = SurfacePublication(
            id="publication-university",
            organization_id=run.organization_id,
            user_id=run.user_id,
            event_id=run.event_id,
            project_id=project.id,
            revision_id=revision.id,
            revision_number=1,
            build_id=build.id,
            action="publish",
            actor_id="surface-builder",
            run_id="a-different-run",
            idempotency_key="publication-idempotency",
            request_fingerprint="publication-fingerprint",
        )
        project.published_revision_id = revision.id
        project.published_build_id = build.id
        session.add_all([project, revision, build, publication])
        await session.commit()

        assert await verified_surface_publication(session, run=run) is None
        guard = SurfaceBuilderCompletionGuard(
            run_context=authenticated_run_context(
                user_id=run.user_id,
                organization_id=run.organization_id,
                run_id=run.id,
                session_id=run.session_id,
                event_id=run.event_id,
                workspace_id=run.event_id,
                agent_id=run.agent_id,
            ),
            session_factory=db_session_maker,
        )
        with pytest.raises(ValueError, match="no verified live publication"):
            await guard.require_publication()
        publication.run_id = run.id
        session.add(publication)
        await session.commit()
        receipt = await verified_surface_publication(session, run=run)
        assert receipt == {
            "project_id": project.id,
            "publication_id": publication.id,
            "revision_id": revision.id,
            "build_id": build.id,
        }
        assert await guard.require_publication() == receipt


async def test_worker_executes_builder_with_only_scoped_surface_capabilities(
    db_session_maker,
    monkeypatch,
):
    monkeypatch.setattr(background_module, "async_session", db_session_maker)
    captured: dict = {}
    scoped_files = object()
    authoring_handler = object()
    build_handler = object()
    assignment = _builder_assignment()

    async def fake_authoring_runtime(*_args, **_kwargs):
        return SimpleNamespace(
            file_backend=scoped_files,
            tool_context_extra={
                SURFACE_AUTHORING_CONTEXT_KEY: authoring_handler,
                SURFACE_BUILD_CONTEXT_KEY: build_handler,
            },
        )

    class FakeOrchestrator:
        async def execute_task(self, **kwargs):
            captured["execution"] = kwargs
            return {
                "success": True,
                "steps_taken": 4,
                "agent": None,
                "result": {"metrics": None},
                "trace": {"execution_receipts": []},
            }

    def fake_orchestrator(**kwargs):
        captured["orchestrator"] = kwargs
        return FakeOrchestrator()

    async def fake_receipt(*_args, **_kwargs):
        return {
            "project_id": "surface-university",
            "publication_id": "publication-university",
            "revision_id": "revision-university",
            "build_id": "build-university",
        }

    async def forbidden_general_surface(*_args, **_kwargs):
        raise AssertionError("builder must not resolve ordinary agent capabilities")

    monkeypatch.setattr(
        background_module,
        "resolve_surface_authoring_runtime",
        fake_authoring_runtime,
    )
    monkeypatch.setattr(background_module, "build_orchestrator", fake_orchestrator)
    monkeypatch.setattr(
        background_module,
        "verified_surface_publication",
        fake_receipt,
    )
    monkeypatch.setattr(
        background_module,
        "resolve_run_surface",
        forbidden_general_surface,
    )

    async with db_session_maker() as session:
        session.add_all(
            [
                Organization(
                    id="org-builder",
                    name="Builder org",
                    slug="builder-org",
                    created_by="alice",
                    policy={},
                ),
                OrganizationMembership(
                    organization_id="org-builder",
                    user_id="alice",
                    role="member",
                ),
                Event(
                    id="evt-builder",
                    organization_id="org-builder",
                    user_id="alice",
                    title="University",
                ),
            ]
        )
        run = Run(
            organization_id="org-builder",
            user_id="alice",
            event_id="evt-builder",
            agent_id="surface-builder",
            session_id="evt-builder",
            run_kind=SURFACE_BUILDER_RUN_KIND,
            run_profile=SURFACE_BUILDER_RUN_PROFILE.descriptor(),
            model_assignment=assignment.descriptor(),
            task="Build the timetable Surface",
            status="running",
            attempt_count=1,
            lease_owner="worker-a",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        session.add(run)
        await session.commit()
        run_id = run.id

    await background_module.execute_claimed_run(run_id, "worker-a")

    assert captured["orchestrator"]["run_profile"] == SURFACE_BUILDER_RUN_PROFILE
    assert captured["orchestrator"]["llm_config"] == assignment.llm_config()
    assert "agent_config" not in captured["orchestrator"]
    assert captured["orchestrator"]["file_backend"] is scoped_files
    builder_context = captured["execution"]["tool_context_extra"]
    assert builder_context[SURFACE_AUTHORING_CONTEXT_KEY] is authoring_handler
    assert builder_context[SURFACE_BUILD_CONTEXT_KEY] is build_handler
    assert isinstance(
        builder_context[SURFACE_COMPLETION_CONTEXT_KEY],
        SurfaceBuilderCompletionGuard,
    )
    assert captured["execution"]["mcp_servers"] == []
    async with db_session_maker() as session:
        completed = await session.get(Run, run_id)
        assert completed is not None
        assert completed.status == "completed"
        assert completed.final_answer == (
            "Your Event Surface is ready. Open it beside this conversation to "
            "use the new visual workspace."
        )
        assert completed.metrics is not None
        assert completed.metrics["aloy_model_assignment"] == assignment.descriptor()
        assert completed.execution_receipts == [
            {
                "kind": "model_assignment",
                **assignment.descriptor(),
            },
            {
                "kind": "surface_publication",
                "project_id": "surface-university",
                "publication_id": "publication-university",
                "revision_id": "revision-university",
                "build_id": "build-university",
            },
        ]
