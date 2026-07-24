from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
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
from aloy_backend.surface_build_runner import SurfaceBuildRunnerResult
from aloy_backend.surface_pipeline import (
    SurfaceCandidate,
    SurfaceCandidateEditEnvelope,
    SurfaceCandidateEnvelope,
    SurfacePipelineResult,
)
from aloy_backend.surface_requests import SURFACE_BUILDER_RUN_KIND
from pori import stable_fingerprint
from pori.llm.messages import ToolCall, ToolTurn


async def test_progress_heartbeat_stop_handles_pre_start_cancellation() -> None:
    started = False

    async def heartbeat() -> None:
        nonlocal started
        started = True
        await asyncio.sleep(60)

    task = asyncio.create_task(heartbeat())
    await builder_module._stop_progress_heartbeat(task)

    assert not started
    assert task.cancelled()


def test_repair_diagnostics_are_compacted_without_losing_compositions() -> None:
    diagnostics = [
        {
            "stage": "preview",
            "code": "state_region_missing",
            "severity": "error",
            "message": "No visible SDK-bound resource region",
            "viewport": viewport,
            "state": state,
        }
        for viewport in ("wide", "mobile")
        for state in ("loading", "empty", "error")
    ]

    compact = builder_module._compact_repair_diagnostics(diagnostics)

    assert len(compact) == 1
    assert compact[0]["occurrences"] == 6
    assert compact[0]["viewports"] == ["wide", "mobile"]
    assert compact[0]["states"] == ["loading", "empty", "error"]


def test_prompt_context_keeps_draft_once_and_compacts_operational_trail() -> None:
    source = "export default function App(){return <main>Career</main>}"
    rendered = json.loads(
        builder_module._render_prompt_context(
            {
                "event": {"id": "evt-career", "title": "Career OS"},
                "brief": {"purpose": "Run a focused job search"},
                "tasks": [],
                "proposals": [],
                "files": [{"id": "file-cv", "name": "cv.pdf"}],
                "file_excerpts": {"/files/notes.txt": "x" * 20_000},
                "trail": [
                    {
                        "id": "trail-1",
                        "kind": "surface_build_failed",
                        "summary": "A candidate was rejected",
                        "created_at": "2026-07-22T00:00:00Z",
                        "payload": {"internal": "x" * 20_000},
                    }
                ],
                "surface": {
                    "expected_revision": "srev-draft",
                    "draft": {"id": "srev-draft", "files": {"/src/App.tsx": source}},
                    "published": {
                        "id": "srev-live",
                        "files": {"/src/App.tsx": source},
                    },
                },
            }
        )
    )

    assert rendered["surface"]["draft"]["files"]["/src/App.tsx"] == source
    assert rendered["surface"]["published"] == {"id": "srev-live"}
    assert rendered["trail"] == [
        {
            "id": "trail-1",
            "kind": "surface_build_failed",
            "summary": "A candidate was rejected",
            "created_at": "2026-07-22T00:00:00Z",
        }
    ]
    assert len(rendered["file_excerpts"]["/files/notes.txt"]) == 8_000


def test_repair_prompt_uses_exact_rejected_source_without_stale_event_context() -> None:
    previous = SurfaceCandidateEditEnvelope.model_validate(
        {
            "summary": "Add resources",
            "changes": [
                {
                    "path": "/workspace/src/App.tsx",
                    "operation": "replace_text",
                    "match": "Old",
                    "replacement": "Resources",
                }
            ],
        }
    )

    messages = builder_module._messages(
        task="Add Event resources",
        context='{"surface":{"draft":{"files":{"/src/App.tsx":"STALE"}}}}',
        instructions="Build safely",
        previous_candidate=previous,
        diagnostics=[
            {
                "stage": "build",
                "code": "typescript_contract_error",
                "message": "Cannot find name filesResource",
            }
        ],
        repair_files={
            "/surface.json": '{"entrypoint":"/src/App.tsx"}',
            "/src/App.tsx": "const current = 'EXACT_REJECTED_SOURCE';",
        },
        candidate_mode="edit",
        required_primary_jobs=[],
    )

    content = messages[-1].content
    assert "EXACT_REJECTED_SOURCE" in content
    assert "Cannot find name filesResource" in content
    assert "STALE" not in content


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


def _provider_candidate(version: int) -> SurfaceCandidateEnvelope:
    candidate = _candidate(version)
    return SurfaceCandidateEnvelope.model_validate(
        {
            "summary": candidate.summary,
            "files": [item.model_dump(mode="python") for item in candidate.files],
        }
    )


async def _create_builder_run(db_session_maker, *, run_id: str) -> None:
    primary_jobs = [
        {
            "id": "job_" + hashlib.sha256(b"See this week").hexdigest()[:16],
            "description": "See this week",
        }
    ]
    contract_body = {
        "policy_version": "aloy-surface-primary-jobs@1",
        "jobs": primary_jobs,
    }
    contract_fingerprint = hashlib.sha256(
        json.dumps(contract_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
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
                    execution_receipts=[
                        {
                            "kind": "surface_primary_job_contract",
                            **contract_body,
                            "fingerprint": contract_fingerprint,
                        }
                    ],
                ),
            ]
        )
        await session.commit()


async def test_default_builder_uses_git_workspace_tools_without_structured_output(
    db_session_maker,
    monkeypatch,
):
    monkeypatch.setattr(builder_module, "async_session", db_session_maker)
    await _create_builder_run(db_session_maker, run_id="builder-workspace")

    class WorkspaceModel:
        model = "accounts/fireworks/models/kimi-k2p6"
        last_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 80,
            "total_tokens": 180,
        }

        async def ainvoke_tools(self, _messages, _tools):
            return ToolTurn(
                tool_calls=[
                    ToolCall(
                        id="manifest",
                        name="write_file",
                        arguments={
                            "path": "/surface.json",
                            "content": json.dumps(
                                {
                                    "format": "aloy-react-surface",
                                    "entrypoint": "/src/App.tsx",
                                    "sdk_version": "1",
                                    "capabilities": [],
                                    "intents": {},
                                    "widgets": [],
                                    "primary_jobs": [
                                        {
                                            "id": "job_"
                                            + hashlib.sha256(
                                                b"See this week"
                                            ).hexdigest()[:16],
                                            "description": "See this week",
                                            "assertions": [
                                                {
                                                    "kind": "visible",
                                                    "role": "heading",
                                                    "name": "Week",
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ),
                        },
                    ),
                    ToolCall(
                        id="app",
                        name="write_file",
                        arguments={
                            "path": "/src/App.tsx",
                            "content": (
                                "export default function App() { return "
                                "<main>Workspace Surface</main> }"
                            ),
                        },
                    ),
                    ToolCall(
                        id="finish",
                        name="finish_candidate",
                        arguments={"summary": "Build in a workspace"},
                    ),
                ]
            )

    class WorkspaceRunner:
        toolchain_version = "test@1"

        async def build(self, *, build_id, files, manifest):
            del build_id, manifest
            assert "Workspace Surface" in files["/src/App.tsx"]
            return SurfaceBuildRunnerResult(status="succeeded", bundle=b"bundle")

    async def runtime_resolver(*_args, **_kwargs):
        return SimpleNamespace(
            prompt_context={
                "event": {"title": "University"},
                "brief": {"summary": "Manage Semester 2"},
                "surface": {"draft": None},
            },
            project_snapshot={"expected_revision": None, "draft": None},
            authoring_handler=object(),
            build_handler=object(),
            workspace_build_runner=WorkspaceRunner(),
        )

    class PublishingPipeline:
        async def execute(self, candidate, *, submission):
            assert submission == 1
            assert any("Workspace Surface" in item.content for item in candidate.files)
            return SurfacePipelineResult(
                status="published",
                candidate_fingerprint=candidate.fingerprint,
                revision_id="revision-workspace",
                build_id="build-workspace",
                publication={"id": "publication-workspace"},
            )

    async def verified_receipt(*_args, **_kwargs):
        return {
            "project_id": "project-1",
            "publication_id": "publication-workspace",
            "revision_id": "revision-workspace",
            "build_id": "build-workspace",
        }

    monkeypatch.setattr(
        builder_module, "verified_surface_publication", verified_receipt
    )
    assert await builder_module.execute_claimed_surface_builder(
        "builder-workspace",
        "worker-a",
        llm_factory=lambda _config: WorkspaceModel(),
        runtime_resolver=runtime_resolver,
        pipeline_factory=lambda **_kwargs: PublishingPipeline(),
    )

    async with db_session_maker() as session:
        run = await session.get(Run, "builder-workspace")
        assert run is not None
        assert run.status == "completed"
        assert run.metrics["structured_output"] is False
        assert run.metrics["tool_calls"] == 3
        receipt = next(
            item
            for item in run.execution_receipts
            if item.get("kind") == "surface_development_workspace"
        )
        assert receipt["protocol"] == "native_tools"
        assert receipt["changed_paths"] == ["src/App.tsx", "surface.json"]


async def test_workspace_repair_keeps_prior_changes_after_pre_persistence_rejection(
    db_session_maker,
    monkeypatch,
):
    """A rejection before any revision persists must not rebase the next edit
    envelope onto the original source and silently drop the first submission's
    changes — the exact failure mode that made Surface updates unrepairable."""
    monkeypatch.setattr(builder_module, "async_session", db_session_maker)
    await _create_builder_run(db_session_maker, run_id="builder-repair-base")

    job_id = "job_" + hashlib.sha256(b"See this week").hexdigest()[:16]
    base_manifest = json.dumps(
        {
            "format": "aloy-react-surface",
            "entrypoint": "/src/App.tsx",
            "sdk_version": "1",
            "capabilities": [],
            "intents": {},
            "widgets": [],
            "interaction_checks": [],
            "primary_jobs": [
                {
                    "id": job_id,
                    "description": "See this week",
                    "assertions": [
                        {"kind": "visible", "role": "heading", "name": "Week"}
                    ],
                }
            ],
        }
    )
    base_files = {
        "/surface.json": base_manifest,
        "/src/App.tsx": "export default function App() { return <main>Old</main> }",
    }

    class WorkspaceModel:
        model = "accounts/fireworks/models/kimi-k2p6"
        last_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 80,
            "total_tokens": 180,
        }

        def __init__(self) -> None:
            self.first_user_context = ""
            self.turns = [
                ToolTurn(
                    tool_calls=[
                        ToolCall(
                            id="edit",
                            name="replace_text",
                            arguments={
                                "path": "/src/App.tsx",
                                "match": "Old",
                                "replacement": "New",
                            },
                        ),
                        ToolCall(id="check", name="run_typecheck", arguments={}),
                        ToolCall(
                            id="finish-one",
                            name="finish_candidate",
                            arguments={"summary": "Apply the requested change"},
                        ),
                    ]
                ),
                ToolTurn(
                    tool_calls=[
                        ToolCall(
                            id="note",
                            name="write_file",
                            arguments={
                                "path": "/src/Note.ts",
                                "content": "export const note = 'kept';\n",
                            },
                        ),
                        ToolCall(id="recheck", name="run_typecheck", arguments={}),
                        ToolCall(
                            id="finish-two",
                            name="finish_candidate",
                            arguments={"summary": "Repair the rejected candidate"},
                        ),
                    ]
                ),
            ]

        async def ainvoke_tools(self, messages, _tools):
            if not self.first_user_context:
                self.first_user_context = str(messages[-1].content)
            return self.turns.pop(0)

    class WorkspaceRunner:
        toolchain_version = "test@1"

        async def build(self, *, build_id, files, manifest):
            del build_id, manifest, files
            return SurfaceBuildRunnerResult(status="succeeded", bundle=b"bundle")

    async def runtime_resolver(*_args, **_kwargs):
        return SimpleNamespace(
            prompt_context={
                "event": {"title": "University"},
                "brief": {"summary": "Manage Semester 2"},
                "surface": {"draft": {"files": base_files}},
            },
            project_snapshot={
                "expected_revision": "revision-base",
                "draft": {"files": base_files},
            },
            authoring_handler=object(),
            build_handler=object(),
            workspace_build_runner=WorkspaceRunner(),
        )

    class RejectThenInspectPipeline:
        def __init__(self) -> None:
            self.second_candidate: SurfaceCandidate | None = None

        async def execute(self, candidate, *, submission):
            files = {item.source_path: item.content for item in candidate.files}
            if submission == 1:
                assert "New" in files["/src/App.tsx"]
                return SurfacePipelineResult(
                    status="repair_required",
                    candidate_fingerprint=candidate.fingerprint,
                    diagnostics=[
                        {
                            "stage": "validation",
                            "code": "primary_job_manifest_mismatch",
                            "severity": "error",
                            "message": "surface.json must declare the frozen jobs",
                            "path": "/surface.json",
                        }
                    ],
                )
            self.second_candidate = candidate
            return SurfacePipelineResult(
                status="published",
                candidate_fingerprint=candidate.fingerprint,
                revision_id="revision-repaired",
                build_id="build-repaired",
                publication={"id": "publication-repaired"},
            )

    pipeline = RejectThenInspectPipeline()

    async def verified_receipt(*_args, **_kwargs):
        return {
            "project_id": "project-1",
            "publication_id": "publication-repaired",
            "revision_id": "revision-repaired",
            "build_id": "build-repaired",
        }

    monkeypatch.setattr(
        builder_module, "verified_surface_publication", verified_receipt
    )
    workspace_model = WorkspaceModel()
    assert await builder_module.execute_claimed_surface_builder(
        "builder-repair-base",
        "worker-a",
        llm_factory=lambda _config: workspace_model,
        runtime_resolver=runtime_resolver,
        pipeline_factory=lambda **_kwargs: pipeline,
    )

    # The workspace owns the source: the prompt keeps file paths but never
    # embeds file bodies, and every call carries the Run's affinity key.
    assert "/src/App.tsx" in workspace_model.first_user_context
    assert "export default function App" not in workspace_model.first_user_context
    assert workspace_model.session_affinity == "builder-repair-base"

    assert pipeline.second_candidate is not None
    repaired = {
        item.source_path: item.content for item in pipeline.second_candidate.files
    }
    assert "New" in repaired["/src/App.tsx"]
    assert repaired["/src/Note.ts"] == "export const note = 'kept';\n"

    async with db_session_maker() as session:
        run = await session.get(Run, "builder-repair-base")
        assert run is not None
        assert run.status == "completed"


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
        assert "tools" not in kwargs
        assert len(messages) == 2
        generated.append(len(generated) + 1)
        if len(generated) == 1:
            assert schema is SurfaceCandidateEnvelope
            return {"parsed": _provider_candidate(1), "raw": {}}
        assert schema is SurfaceCandidateEditEnvelope
        return {
            "parsed": SurfaceCandidateEditEnvelope.model_validate(
                {
                    "summary": "University workspace v2",
                    "changes": [
                        {
                            "path": "/workspace/src/App.tsx",
                            "operation": "write",
                            "content": (
                                "export default function App() { return "
                                "<main>v2</main> }"
                            ),
                        }
                    ],
                }
            ),
            "raw": {},
        }

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


async def test_builder_streams_incremental_edits_for_existing_surface(
    db_session_maker,
    monkeypatch,
):
    monkeypatch.setattr(builder_module, "async_session", db_session_maker)
    await _create_builder_run(db_session_maker, run_id="builder-incremental")
    llm = SimpleNamespace(
        last_usage={"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}
    )
    manifest = (
        '{"format":"aloy-react-surface","entrypoint":"/src/App.tsx",'
        '"sdk_version":"1","capabilities":[],"intents":{},"widgets":[]}'
    )
    app_source = "export default function App(){return <main>Old</main>}"
    factory_arguments: list[dict] = []

    async def structured_invoker(_llm, schema, messages, **kwargs):
        assert schema is SurfaceCandidateEditEnvelope
        assert "Revision contract" in messages[-1].content
        kwargs["on_delta"]('{"changes":[')
        return {
            "parsed": SurfaceCandidateEditEnvelope.model_validate(
                {
                    "summary": "Career resources",
                    "changes": [
                        {
                            "path": "/workspace/src/App.tsx",
                            "operation": "write",
                            "content": (
                                "export default function App(){return "
                                "<main>Resources</main>}"
                            ),
                        }
                    ],
                }
            ),
            "raw": None,
        }

    async def runtime_resolver(*_args, **_kwargs):
        snapshot = {
            "expected_revision": "revision-live",
            "draft": {
                "id": "revision-live",
                "files": {
                    "/surface.json": manifest,
                    "/src/App.tsx": app_source,
                },
            },
        }
        return SimpleNamespace(
            project_snapshot=snapshot,
            prompt_context={"event": {"title": "Career"}, "surface": snapshot},
            authoring_handler=object(),
            build_handler=object(),
        )

    class PublishingPipeline:
        async def execute(self, candidate, *, submission):
            files = {item.source_path: item.content for item in candidate.files}
            assert submission == 1
            assert files["/surface.json"] == manifest
            assert "Resources" in files["/src/App.tsx"]
            return SurfacePipelineResult(
                status="published",
                candidate_fingerprint=candidate.fingerprint,
                revision_id="revision-new",
                build_id="build-new",
                publication={"id": "publication-new"},
            )

    def pipeline_factory(**kwargs):
        factory_arguments.append(kwargs)
        return PublishingPipeline()

    async def verified_receipt(*_args, **_kwargs):
        return {
            "project_id": "project-1",
            "publication_id": "publication-new",
            "revision_id": "revision-new",
            "build_id": "build-new",
        }

    monkeypatch.setattr(
        builder_module, "verified_surface_publication", verified_receipt
    )
    assert await builder_module.execute_claimed_surface_builder(
        "builder-incremental",
        "worker-a",
        llm_factory=lambda _config: llm,
        structured_invoker=structured_invoker,
        runtime_resolver=runtime_resolver,
        pipeline_factory=pipeline_factory,
    )

    assert factory_arguments[0]["expected_base_revision_id"] == "revision-live"
    async with db_session_maker() as session:
        run = await session.get(Run, "builder-incremental")
        assert run is not None
        receipt = next(
            item
            for item in run.execution_receipts or []
            if item.get("kind") == "surface_candidate"
        )
        assert receipt["candidate_mode"] == "edit"
        assert receipt["changed_file_count"] == 1
        assert receipt["generation_phase"] == "receiving_output"
        assert receipt["output_chars"] > 0


async def test_generation_timeout_cancels_stalled_provider_call():
    progress = builder_module._GenerationProgress()

    async def stalled():
        await asyncio.sleep(60)

    with pytest.raises(builder_module.SurfaceGenerationTimeoutError):
        await builder_module._await_candidate_generation(
            stalled(),
            progress=progress,
            first_output_timeout_seconds=0.01,
            deadline=builder_module.perf_counter() + 1,
        )


async def test_turn_boundary_silence_within_generation_timeout_is_not_a_stall(
    monkeypatch,
):
    """Workspace turns are non-streaming provider calls that report progress
    only when they complete; silence shorter than the declared per-call
    generation timeout must not be treated as a stalled stream."""
    monkeypatch.setattr(builder_module, "SURFACE_STREAM_IDLE_TIMEOUT_SECONDS", 0.05)
    progress = builder_module._GenerationProgress()
    progress.on_delta("workspace turn 1; tool calls 2\n")

    async def slow_turn():
        await asyncio.sleep(0.2)
        return {"parsed": "candidate"}

    result = await builder_module._await_candidate_generation(
        slow_turn(),
        progress=progress,
        first_output_timeout_seconds=0.5,
        deadline=builder_module.perf_counter() + 5,
    )
    assert result == {"parsed": "candidate"}


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
            invalid = _provider_candidate(1).model_dump(mode="python")
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
                _provider_candidate(2).model_dump(mode="python")
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
        return {"parsed": _provider_candidate(1), "raw": {}}

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
        assert (
            len(
                [
                    item
                    for item in run.execution_receipts or []
                    if item.get("kind") == "surface_candidate"
                ]
            )
            == 1
        )
        assert (
            len(
                [
                    item
                    for item in run.execution_receipts or []
                    if item.get("kind") == "surface_candidate_duplicate"
                ]
            )
            == 1
        )
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


async def test_builder_never_opens_a_fourth_paid_diagnostic_cycle(
    db_session_maker,
    monkeypatch,
):
    monkeypatch.setattr(builder_module, "async_session", db_session_maker)
    await _create_builder_run(db_session_maker, run_id="builder-one-repair")
    llm = SimpleNamespace(last_usage={})
    generated = 0

    async def structured_invoker(_llm, schema, _messages, **_kwargs):
        nonlocal generated
        generated += 1
        if generated == 1:
            assert schema is SurfaceCandidateEnvelope
            return {"parsed": _provider_candidate(1), "raw": {}}
        assert schema is SurfaceCandidateEditEnvelope
        previous = generated - 1
        return {
            "parsed": SurfaceCandidateEditEnvelope.model_validate(
                {
                    "summary": f"University workspace repair {previous}",
                    "changes": [
                        {
                            "path": "/workspace/src/App.tsx",
                            "operation": "replace_text",
                            "match": f"v{previous}",
                            "replacement": f"v{generated}",
                        }
                    ],
                }
            ),
            "raw": {},
        }

    async def runtime_resolver(*_args, **_kwargs):
        return SimpleNamespace(
            prompt_context={"event": {}, "surface": {}},
            authoring_handler=object(),
            build_handler=object(),
        )

    class SerialRejectingPipeline:
        async def execute(self, candidate, *, submission):
            return SurfacePipelineResult(
                status="repair_required",
                candidate_fingerprint=candidate.fingerprint,
                revision_id=f"revision-{submission}",
                build_id=f"build-{submission}",
                diagnostics=[
                    {
                        "stage": "preview",
                        "code": f"quality_failure_{submission}",
                        "severity": "error",
                        "message": "A later quality class also failed",
                    }
                ],
            )

    assert await builder_module.execute_claimed_surface_builder(
        "builder-one-repair",
        "worker-a",
        llm_factory=lambda _config: llm,
        structured_invoker=structured_invoker,
        runtime_resolver=runtime_resolver,
        pipeline_factory=lambda **_kwargs: SerialRejectingPipeline(),
    )

    assert generated == 3
    async with db_session_maker() as session:
        run = await session.get(Run, "builder-one-repair")
        assert run is not None
        assert run.status == "failed"
        assert run.steps_taken == 3
        assert run.progress is not None
        assert run.progress["submission"] == 3
        assert run.progress["submissions"] == 3
        candidates = [
            item
            for item in run.execution_receipts or []
            if item.get("kind") == "surface_candidate"
        ]
        assert [item["submission"] for item in candidates] == [1, 2, 3]
        assert not any(
            item.get("kind") == "surface_publication"
            for item in run.execution_receipts or []
        )


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
        return {"parsed": _provider_candidate(1), "raw": {}}

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
