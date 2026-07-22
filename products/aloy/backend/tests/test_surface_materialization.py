from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlmodel import select

from aloy_backend import background as background_module
from aloy_backend import surface_materialization as materialization_module
from aloy_backend.background import execute_claimed_run
from aloy_backend.models import (
    Event,
    EventTrailEntry,
    Run,
    SurfaceProject,
    SurfaceRevision,
)
from aloy_backend.surface_materialization import (
    SURFACE_MATERIALIZATION_RUN_KIND,
    execute_claimed_surface_materialization,
    queue_surface_revision_materialization,
)
from aloy_backend.surface_pipeline import SurfacePipelineResult
from aloy_backend.tenancy import OrganizationPolicy


async def _source_bound_run(client, db_session_maker) -> tuple[str, str]:
    response = await client.post(
        "/v1/events",
        json={"title": "Starting Surface", "summary": "A reviewed starting point"},
    )
    assert response.status_code == 201
    event_id = response.json()["id"]
    async with db_session_maker() as session:
        event = await session.get(Event, event_id)
        assert event is not None
        project = SurfaceProject(
            organization_id=event.organization_id,
            user_id=event.user_id,
            event_id=event.id,
            sdk_version="1",
            lifecycle="draft",
        )
        session.add(project)
        await session.flush()
        revision = SurfaceRevision(
            organization_id=event.organization_id,
            user_id=event.user_id,
            event_id=event.id,
            project_id=project.id,
            revision_number=1,
            idempotency_key="reviewed-source-revision",
            request_fingerprint="b" * 64,
            manifest={
                "format": "aloy-react-surface",
                "entrypoint": "/src/App.tsx",
                "sdk_version": "1",
                "capabilities": [],
                "intents": {},
                "interaction_checks": [],
                "primary_jobs": [],
                "widgets": [],
            },
            files={
                "/src/App.tsx": "export default function App(){return <main>Ready</main>}"
            },
            checksum="a" * 64,
            file_count=1,
            total_bytes=61,
        )
        session.add(revision)
        await session.flush()
        project.draft_revision_id = revision.id
        session.add(project)
        run, replayed = await queue_surface_revision_materialization(
            session,
            event=event,
            project=project,
            revision=revision,
            policy=OrganizationPolicy(),
            trigger="test_reviewed_source",
            actor_id=event.user_id,
            origin_evidence=[{"review_id": "review-1"}],
        )
        assert replayed is False
        await session.commit()
        return event_id, run.id


async def _claim(db_session_maker, run_id: str, worker_id: str, *, attempt: int = 1):
    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        assert run is not None
        run.status = "running"
        run.attempt_count = attempt
        run.lease_owner = worker_id
        run.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
        run.started_at = run.started_at or datetime.now(timezone.utc)
        session.add(run)
        await session.commit()


async def test_reviewed_source_queue_is_model_free_idempotent_and_user_visible(
    client, db_session_maker
):
    event_id, run_id = await _source_bound_run(client, db_session_maker)

    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        event = await session.get(Event, event_id)
        project = (
            (
                await session.execute(
                    select(SurfaceProject).where(SurfaceProject.event_id == event_id)
                )
            )
            .scalars()
            .one()
        )
        revision = await session.get(SurfaceRevision, project.draft_revision_id)
        assert run is not None and event is not None and revision is not None
        assert run.run_kind == SURFACE_MATERIALIZATION_RUN_KIND
        assert run.agent_id == "surface-host"
        assert run.conversation_id is None
        assert run.model_assignment is None
        assert run.run_profile["model_tools"] == []
        assert run.run_profile["revision_id"] == revision.id
        replay, replayed = await queue_surface_revision_materialization(
            session,
            event=event,
            project=project,
            revision=revision,
            policy=OrganizationPolicy(),
            trigger="test_reviewed_source",
            actor_id=event.user_id,
        )
        assert replayed is True
        assert replay.id == run.id

    status = await client.get(f"/v1/events/{event_id}/surface/status")
    assert status.status_code == 200
    assert status.json()["run_id"] == run_id
    assert status.json()["stage"] == "queued"
    assert status.json()["message"] == "Preparing your starting Surface"


async def test_materialization_runs_the_shared_pipeline_and_records_publication(
    client, db_session_maker, monkeypatch
):
    event_id, run_id = await _source_bound_run(client, db_session_maker)
    await _claim(db_session_maker, run_id, "worker-materialize")
    observed: dict = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            self.observe = kwargs["stage_observer"]

        async def execute(self, **kwargs):
            observed.update(kwargs)
            for stage in (
                "building_bundle",
                "inspecting_preview",
                "publishing_surface",
            ):
                await self.observe(stage)
            return SurfacePipelineResult(
                status="published",
                candidate_fingerprint=kwargs["source_fingerprint"],
                revision_id=kwargs["revision_id"],
                build_id="build-reviewed-source",
                publication={"id": "publication-reviewed-source"},
                timings_ms={"build": 1.0, "preview": 2.0, "publish": 1.0},
            )

    async def verified(_session, *, run):
        return {
            "project_id": run.run_profile["project_id"],
            "publication_id": "publication-reviewed-source",
            "revision_id": run.run_profile["revision_id"],
            "build_id": "build-reviewed-source",
        }

    monkeypatch.setattr(
        materialization_module, "verified_surface_publication", verified
    )
    completed = await execute_claimed_surface_materialization(
        run_id,
        "worker-materialize",
        session_factory=db_session_maker,
        handler_factory=lambda **_kwargs: object(),
        pipeline_factory=FakePipeline,
    )

    assert completed is True
    assert observed["attempt"] == 1
    assert observed["source_fingerprint"] == "a" * 64
    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        assert run is not None
        assert run.status == "completed"
        assert run.success is True
        assert run.progress["stage"] == "published"
        assert run.lease_owner is None
        assert any(
            receipt.get("kind") == "surface_materialization_result"
            for receipt in run.execution_receipts
        )
        finished = (
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.event_id == event_id,
                        EventTrailEntry.kind == "surface_materialization_finished",
                    )
                )
            )
            .scalars()
            .one()
        )
        assert finished.payload["status"] == "published"


async def test_host_failure_advances_only_the_pipeline_attempt_before_retry(
    client, db_session_maker, monkeypatch
):
    _, run_id = await _source_bound_run(client, db_session_maker)
    attempts: list[int] = []

    class FlakyPipeline:
        def __init__(self, **_kwargs):
            pass

        async def execute(self, **kwargs):
            attempts.append(kwargs["attempt"])
            if len(attempts) == 1:
                return SurfacePipelineResult(
                    status="host_failed",
                    candidate_fingerprint=kwargs["source_fingerprint"],
                    revision_id=kwargs["revision_id"],
                    diagnostics=[
                        {
                            "stage": "build",
                            "code": "isolated_builder_unavailable",
                            "severity": "error",
                            "message": "Sandbox is temporarily unavailable",
                        }
                    ],
                )
            return SurfacePipelineResult(
                status="published",
                candidate_fingerprint=kwargs["source_fingerprint"],
                revision_id=kwargs["revision_id"],
                build_id="build-retry",
                publication={"id": "publication-retry"},
            )

    async def verified(_session, *, run):
        return {
            "project_id": run.run_profile["project_id"],
            "publication_id": "publication-retry",
            "revision_id": run.run_profile["revision_id"],
            "build_id": "build-retry",
        }

    monkeypatch.setattr(
        materialization_module, "verified_surface_publication", verified
    )
    await _claim(db_session_maker, run_id, "worker-retry", attempt=1)
    await execute_claimed_surface_materialization(
        run_id,
        "worker-retry",
        session_factory=db_session_maker,
        handler_factory=lambda **_kwargs: object(),
        pipeline_factory=FlakyPipeline,
    )
    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        assert run is not None
        assert run.status == "pending"
        assert run.progress["pipeline_attempt"] == 2

    await _claim(db_session_maker, run_id, "worker-retry", attempt=2)
    await execute_claimed_surface_materialization(
        run_id,
        "worker-retry",
        session_factory=db_session_maker,
        handler_factory=lambda **_kwargs: object(),
        pipeline_factory=FlakyPipeline,
    )
    assert attempts == [1, 2]
    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        assert run is not None
        assert run.status == "completed"


async def test_tampered_source_binding_fails_closed_before_the_pipeline(
    client, db_session_maker
):
    _, run_id = await _source_bound_run(client, db_session_maker)
    await _claim(db_session_maker, run_id, "worker-tamper")
    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        assert run is not None
        receipts = [dict(item) for item in run.execution_receipts or []]
        receipts[0]["source_checksum"] = "f" * 64
        run.execution_receipts = receipts
        session.add(run)
        await session.commit()

    touched = False

    def handler_factory(**_kwargs):
        nonlocal touched
        touched = True
        return object()

    completed = await execute_claimed_surface_materialization(
        run_id,
        "worker-tamper",
        session_factory=db_session_maker,
        handler_factory=handler_factory,
    )
    assert completed is True
    assert touched is False
    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        assert run is not None
        assert run.status == "failed"
        assert run.lease_owner is None
        assert "integrity" in run.reasoning


async def test_background_dispatches_materialization_without_general_agent_tools(
    client, db_session_maker, monkeypatch
):
    _, run_id = await _source_bound_run(client, db_session_maker)
    await _claim(db_session_maker, run_id, "worker-dispatch")
    called: list[tuple[str, str]] = []

    async def fake_executor(next_run_id: str, worker_id: str) -> bool:
        called.append((next_run_id, worker_id))
        return True

    monkeypatch.setattr(
        background_module,
        "execute_claimed_surface_materialization",
        fake_executor,
    )
    monkeypatch.setattr(background_module, "async_session", db_session_maker)
    await execute_claimed_run(run_id, "worker-dispatch")
    assert called == [(run_id, "worker-dispatch")]
