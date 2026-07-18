from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from sqlmodel import select

from aloy_backend import surface_builder as builder_module
from aloy_backend.model_roles import ModelAssignment, ModelRole
from aloy_backend.models import (
    Event,
    EventTrailEntry,
    Organization,
    OrganizationMembership,
    Run,
)
from aloy_backend.run_profiles import SURFACE_BUILDER_RUN_PROFILE
from aloy_backend.surface_pipeline import (
    SurfaceCandidate,
    SurfaceCandidateEnvelope,
    SurfacePipelineResult,
)
from aloy_backend.surface_requests import SURFACE_BUILDER_RUN_KIND
from pori import stable_fingerprint


def _assignment() -> ModelAssignment:
    values = {
        "config_version": 1,
        "role": ModelRole.SURFACE_BUILDER,
        "provider": "fireworks",
        "model": "accounts/fireworks/models/kimi-k2p6",
        "temperature": 0.1,
        "max_tokens": 32768,
        "generation_timeout_seconds": 120,
        "reasoning_mode": "none",
        "capabilities": ("structured_output", "tools"),
        "skill_id": "surface-builder@1",
        "qualification_status": "qualified",
        "qualification_suite": "aloy-surface-builder-v1-dev-smoke",
        "qualification_evidence": "temporary local smoke assignment",
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
            "resolution_ms": 1.0,
            "resolved_at": datetime.now(timezone.utc),
        }
    )


def _candidate(version: int) -> SurfaceCandidate:
    return SurfaceCandidate.model_validate(
        {
            "summary": f"University workspace v{version}",
            "primary_jobs": ["See this week"],
            "files": [
                {
                    "path": "/workspace/surface.json",
                    "content": (
                        '{"format":"aloy-react-surface",'
                        '"entrypoint":"/src/App.tsx","sdk_version":"1",'
                        '"capabilities":[],"intents":{},"widgets":[]}'
                    ),
                },
                {
                    "path": "/workspace/src/App.tsx",
                    "content": f"export default function App() {{ return <main>v{version}</main> }}",
                },
            ],
        }
    )


async def _create_builder_run(db_session_maker, *, run_id: str) -> None:
    async with db_session_maker() as session:
        session.add_all(
            [
                Organization(
                    id=f"org-{run_id}",
                    name="Builder org",
                    slug=f"builder-{run_id}",
                    created_by="alice",
                    policy={},
                ),
                OrganizationMembership(
                    organization_id=f"org-{run_id}",
                    user_id="alice",
                    role="member",
                ),
                Event(
                    id=f"event-{run_id}",
                    organization_id=f"org-{run_id}",
                    user_id="alice",
                    title="University",
                ),
                Run(
                    id=run_id,
                    organization_id=f"org-{run_id}",
                    user_id="alice",
                    event_id=f"event-{run_id}",
                    agent_id="surface-builder",
                    session_id=f"event-{run_id}",
                    run_kind=SURFACE_BUILDER_RUN_KIND,
                    run_profile=SURFACE_BUILDER_RUN_PROFILE.descriptor(),
                    model_assignment=_assignment().descriptor(),
                    task="Return a complete University timetable Surface candidate",
                    status="running",
                    attempt_count=1,
                    lease_owner="worker-a",
                    lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                    started_at=datetime.now(timezone.utc),
                ),
            ]
        )
        await session.commit()


async def test_builder_uses_structured_output_and_host_repairs_without_tools(
    db_session_maker,
    monkeypatch,
):
    monkeypatch.setattr(builder_module, "async_session", db_session_maker)
    await _create_builder_run(db_session_maker, run_id="builder-repair")
    llm = SimpleNamespace(
        last_usage={"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200}
    )
    generated: list[int] = []

    async def structured_invoker(_llm, schema, messages, **kwargs):
        assert schema is SurfaceCandidateEnvelope
        assert "tools" not in kwargs
        assert len(messages) == 2
        generated.append(len(generated) + 1)
        return {"parsed": _candidate(len(generated)), "raw": {}}

    async def runtime_resolver(*_args, **_kwargs):
        return SimpleNamespace(
            prompt_context={
                "event": {"title": "University"},
                "brief": {"summary": "Manage Semester 2"},
                "surface": {"draft": None},
            },
            authoring_handler=object(),
            build_handler=object(),
        )

    class FakePipeline:
        async def execute(self, candidate, *, submission):
            if submission == 1:
                return SurfacePipelineResult(
                    status="repair_required",
                    candidate_fingerprint=candidate.fingerprint,
                    revision_id="revision-1",
                    build_id="build-1",
                    diagnostics=[
                        {
                            "stage": "build",
                            "code": "typescript_error",
                            "severity": "error",
                            "message": "Repair App.tsx",
                        }
                    ],
                    timings_ms={"persist": 2.0, "build": 20.0},
                )
            return SurfacePipelineResult(
                status="published",
                candidate_fingerprint=candidate.fingerprint,
                revision_id="revision-2",
                build_id="build-2",
                publication={"id": "publication-2"},
                timings_ms={
                    "persist": 2.0,
                    "build": 20.0,
                    "preview": 3.0,
                    "publish": 4.0,
                },
            )

    async def verified_receipt(*_args, **_kwargs):
        return {
            "project_id": "project-1",
            "publication_id": "publication-2",
            "revision_id": "revision-2",
            "build_id": "build-2",
        }

    monkeypatch.setattr(
        builder_module, "verified_surface_publication", verified_receipt
    )
    assert await builder_module.execute_claimed_surface_builder(
        "builder-repair",
        "worker-a",
        llm_factory=lambda _config: llm,
        structured_invoker=structured_invoker,
        runtime_resolver=runtime_resolver,
        pipeline_factory=lambda **_kwargs: FakePipeline(),
    )

    assert generated == [1, 2]
    async with db_session_maker() as session:
        run = await session.get(Run, "builder-repair")
        assert run is not None
        assert run.status == "completed"
        assert run.success is True
        assert run.steps_taken == 2
        assert run.metrics is not None
        assert run.metrics["structured_output"] is True
        assert run.metrics["tool_calls"] == 0
        assert run.metrics["tokens"] == {
            "input": 240,
            "output": 160,
            "total": 400,
            "cache_read": 0,
            "cache_write": 0,
        }
        assert run.selected_skills == ["surface-builder@1"]
        candidate_receipts = [
            item
            for item in run.execution_receipts or []
            if item.get("kind") == "surface_candidate"
        ]
        assert [item["status"] for item in candidate_receipts] == [
            "repair_required",
            "published",
        ]


async def test_builder_repairs_host_source_contract_before_pipeline(
    db_session_maker,
    monkeypatch,
):
    monkeypatch.setattr(builder_module, "async_session", db_session_maker)
    await _create_builder_run(db_session_maker, run_id="builder-contract-repair")
    llm = SimpleNamespace(
        last_usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    )
    generated = 0

    async def structured_invoker(_llm, schema, messages, **kwargs):
        nonlocal generated
        assert schema is SurfaceCandidateEnvelope
        generated += 1
        if generated == 1:
            invalid = _candidate(1).model_dump(mode="python")
            invalid["files"].append(
                {
                    "path": "/workspace/index.html",
                    "content": "<html><body><div id='root'></div></body></html>",
                }
            )
            return {
                "parsed": SurfaceCandidateEnvelope.model_validate(invalid),
                "raw": None,
            }
        assert "index.html" in messages[-1].content
        assert "Unsupported Surface source extension" in messages[-1].content
        return {
            "parsed": SurfaceCandidateEnvelope.model_validate(
                _candidate(2).model_dump(mode="python")
            ),
            "raw": None,
        }

    async def runtime_resolver(*_args, **_kwargs):
        return SimpleNamespace(
            prompt_context={"event": {"title": "University"}, "surface": {}},
            authoring_handler=object(),
            build_handler=object(),
        )

    class PublishingPipeline:
        async def execute(self, candidate, *, submission):
            assert submission == 2
            assert all(item.source_path != "/index.html" for item in candidate.files)
            return SurfacePipelineResult(
                status="published",
                candidate_fingerprint=candidate.fingerprint,
                revision_id="revision-2",
                build_id="build-2",
                publication={"id": "publication-2"},
            )

    async def verified_receipt(*_args, **_kwargs):
        return {
            "project_id": "project-1",
            "publication_id": "publication-2",
            "revision_id": "revision-2",
            "build_id": "build-2",
        }

    monkeypatch.setattr(
        builder_module, "verified_surface_publication", verified_receipt
    )
    assert await builder_module.execute_claimed_surface_builder(
        "builder-contract-repair",
        "worker-a",
        llm_factory=lambda _config: llm,
        structured_invoker=structured_invoker,
        runtime_resolver=runtime_resolver,
        pipeline_factory=lambda **_kwargs: PublishingPipeline(),
    )

    async with db_session_maker() as session:
        run = await session.get(Run, "builder-contract-repair")
        assert run is not None
        assert run.status == "completed"
        assert run.steps_taken == 2
        receipts = run.execution_receipts or []
        contract_rejections = [
            item
            for item in receipts
            if item.get("kind") == "surface_candidate_contract_rejected"
        ]
        assert len(contract_rejections) == 1
        assert contract_rejections[0]["diagnostics"][0]["path"] == "files.2.path"
        assert any(
            item.get("kind") == "surface_candidate"
            and item.get("status") == "published"
            for item in receipts
        )


async def test_builder_exhaustion_fails_without_claiming_publication(
    db_session_maker,
    monkeypatch,
):
    monkeypatch.setattr(builder_module, "async_session", db_session_maker)
    await _create_builder_run(db_session_maker, run_id="builder-exhausted")
    llm = SimpleNamespace(last_usage={})

    async def structured_invoker(*_args, **_kwargs):
        return {"parsed": _candidate(1), "raw": {}}

    async def runtime_resolver(*_args, **_kwargs):
        return SimpleNamespace(
            prompt_context={"event": {}, "surface": {}},
            authoring_handler=object(),
            build_handler=object(),
        )

    class RejectingPipeline:
        async def execute(self, candidate, *, submission):
            return SurfacePipelineResult(
                status="repair_required",
                candidate_fingerprint=candidate.fingerprint,
                diagnostics=[
                    {
                        "stage": "build",
                        "code": "invalid",
                        "severity": "error",
                        "message": "Still invalid",
                    }
                ],
            )

    assert await builder_module.execute_claimed_surface_builder(
        "builder-exhausted",
        "worker-a",
        llm_factory=lambda _config: llm,
        structured_invoker=structured_invoker,
        runtime_resolver=runtime_resolver,
        pipeline_factory=lambda **_kwargs: RejectingPipeline(),
    )

    async with db_session_maker() as session:
        run = await session.get(Run, "builder-exhausted")
        assert run is not None
        assert run.status == "failed"
        assert run.success is False
        assert run.steps_taken == 2
        assert not any(
            item.get("kind") == "surface_publication"
            for item in run.execution_receipts or []
        )
        trails = list(
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.run_id == run.id,
                        EventTrailEntry.kind == "surface_build_failed",
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(trails) == 1


async def test_builder_does_not_spend_model_repair_on_host_failure(
    db_session_maker,
    monkeypatch,
):
    monkeypatch.setattr(builder_module, "async_session", db_session_maker)
    await _create_builder_run(db_session_maker, run_id="builder-host-failed")
    llm = SimpleNamespace(last_usage={})
    generated = 0

    async def structured_invoker(*_args, **_kwargs):
        nonlocal generated
        generated += 1
        return {"parsed": _candidate(1), "raw": {}}

    async def runtime_resolver(*_args, **_kwargs):
        return SimpleNamespace(
            prompt_context={"event": {}, "surface": {}},
            authoring_handler=object(),
            build_handler=object(),
        )

    class HostFailingPipeline:
        async def execute(self, candidate, *, submission):
            assert submission == 1
            return SurfacePipelineResult(
                status="host_failed",
                candidate_fingerprint=candidate.fingerprint,
                revision_id="revision-retained",
                build_id="build-storage-failed",
                diagnostics=[
                    {
                        "stage": "build",
                        "code": "bundle_storage_failed",
                        "severity": "error",
                        "message": "The host object store was unavailable",
                    }
                ],
            )

    assert await builder_module.execute_claimed_surface_builder(
        "builder-host-failed",
        "worker-a",
        llm_factory=lambda _config: llm,
        structured_invoker=structured_invoker,
        runtime_resolver=runtime_resolver,
        pipeline_factory=lambda **_kwargs: HostFailingPipeline(),
    )

    assert generated == 1
    async with db_session_maker() as session:
        run = await session.get(Run, "builder-host-failed")
        assert run is not None
        assert run.status == "failed"
        assert run.steps_taken == 1
        assert "trusted host" in run.final_answer
        assert run.progress["diagnostic"]["code"] == "bundle_storage_failed"
        candidates = [
            item
            for item in run.execution_receipts or []
            if item.get("kind") == "surface_candidate"
        ]
        assert [item["status"] for item in candidates] == ["host_failed"]


async def test_builder_retains_rejected_provider_output_and_usage(
    db_session_maker,
    monkeypatch,
):
    monkeypatch.setattr(builder_module, "async_session", db_session_maker)
    await _create_builder_run(db_session_maker, run_id="builder-invalid-output")
    async with db_session_maker() as session:
        run = await session.get(Run, "builder-invalid-output")
        assert run is not None
        # Schema-invalid output is deterministic for the frozen assignment and
        # must fail closed instead of repeating the same expensive model call.
        run.attempt_count = 1
        run.execution_receipts = [
            {
                "kind": "surface_candidate_rejected",
                "worker_attempt": 2,
                "diagnostic": {"error": "earlier rejection"},
            }
        ]
        session.add(run)
        await session.commit()

    llm = SimpleNamespace(
        last_usage={"prompt_tokens": 90, "completion_tokens": 30, "total_tokens": 120}
    )
    raw = '{"files":[{"path":"/workspace/src/App.tsx","content":"export default App"}]'

    async def structured_invoker(*_args, **_kwargs):
        return {
            "parsed": None,
            "raw": raw,
            "error": "Invalid JSON: unexpected end of input",
        }

    async def runtime_resolver(*_args, **_kwargs):
        return SimpleNamespace(
            prompt_context={"event": {}, "surface": {}},
            authoring_handler=object(),
            build_handler=object(),
        )

    assert await builder_module.execute_claimed_surface_builder(
        "builder-invalid-output",
        "worker-a",
        llm_factory=lambda _config: llm,
        structured_invoker=structured_invoker,
        runtime_resolver=runtime_resolver,
        pipeline_factory=lambda **_kwargs: SimpleNamespace(),
    )

    async with db_session_maker() as session:
        run = await session.get(Run, "builder-invalid-output")
        assert run is not None
        assert run.status == "failed"
        assert run.reasoning == "Invalid JSON: unexpected end of input"
        assert run.metrics is not None
        assert run.metrics["tokens"]["total"] == 120
        rejected = [
            receipt
            for receipt in run.execution_receipts or []
            if receipt.get("kind") == "surface_candidate_rejected"
        ]
        assert len(rejected) == 2
        diagnostic = rejected[-1]["diagnostic"]
        assert diagnostic["error"] == "Invalid JSON: unexpected end of input"
        assert diagnostic["raw_chars"] == len(raw)
        assert diagnostic["raw_head"] == raw
        assert diagnostic["raw_tail"] == ""
        assert diagnostic["raw_truncated"] is False
        assert diagnostic["contains_react_source"] is True
        assert len(diagnostic["raw_sha256"]) == 64
        assert run.progress is not None
        assert run.progress["diagnostic"]["raw_chars"] == len(raw)
        assert "raw_head" not in run.progress["diagnostic"]
