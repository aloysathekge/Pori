from __future__ import annotations

import pytest
from pydantic import ValidationError

from aloy_backend.surface_authoring import SurfaceAuthoringError, SurfaceConflictError
from aloy_backend.surface_pipeline import (
    SurfaceCandidate,
    SurfaceCandidateEditEnvelope,
    SurfaceHostPipeline,
    SurfaceRevisionHostPipeline,
    bind_surface_manifest_primary_jobs,
    materialize_surface_candidate_edit,
)

pytestmark = pytest.mark.asyncio


def _candidate(label: str = "University") -> SurfaceCandidate:
    return SurfaceCandidate.model_validate(
        {
            "summary": f"A useful {label} workspace",
            "primary_jobs": ["See the week", "Open a course"],
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
                    "content": f"export default function App() {{ return <main>{label}</main> }}",
                },
            ],
        }
    )


async def test_incremental_edit_materializes_complete_candidate_from_frozen_source():
    candidate = materialize_surface_candidate_edit(
        SurfaceCandidateEditEnvelope.model_validate(
            {
                "summary": "Career resources",
                "changes": [
                    {
                        "path": "src/App.tsx",
                        "operation": "write",
                        "content": "export default function App(){return <main>Resources</main>}",
                    },
                    {
                        "path": "/workspace/src/old.css",
                        "operation": "delete",
                        "content": None,
                    },
                ],
            }
        ),
        base_files={
            "/surface.json": (
                '{"format":"aloy-react-surface","entrypoint":"/src/App.tsx",'
                '"sdk_version":"1","capabilities":[],"intents":{},"widgets":[]}'
            ),
            "/src/App.tsx": "export default function App(){return <main>Old</main>}",
            "/src/old.css": "main { color: red; }",
        },
        primary_jobs=["See resources"],
    )

    files = {item.source_path: item.content for item in candidate.files}
    assert files["/src/App.tsx"].endswith("Resources</main>}")
    assert "/src/old.css" not in files
    assert "/surface.json" in files


async def test_exact_text_edit_changes_one_fragment_without_rewriting_the_file():
    original = (
        "export default function App(){return "
        "<SurfaceRoot><main>Resources</main></SurfaceRoot>}"
    )
    candidate = materialize_surface_candidate_edit(
        SurfaceCandidateEditEnvelope.model_validate(
            {
                "summary": "Accessible resources",
                "changes": [
                    {
                        "path": "/workspace/src/App.tsx",
                        "operation": "replace_text",
                        "match": "<main>Resources</main>",
                        "replacement": "<section>Resources</section>",
                    }
                ],
            }
        ),
        base_files={
            "/surface.json": (
                '{"format":"aloy-react-surface","entrypoint":"/src/App.tsx",'
                '"sdk_version":"1","capabilities":[],"intents":{},"widgets":[]}'
            ),
            "/src/App.tsx": original,
        },
        primary_jobs=["See resources"],
    )

    files = {item.source_path: item.content for item in candidate.files}
    assert files["/src/App.tsx"] == original.replace(
        "<main>Resources</main>",
        "<section>Resources</section>",
    )


async def test_ordered_exact_text_edits_can_change_the_same_file_atomically():
    candidate = materialize_surface_candidate_edit(
        SurfaceCandidateEditEnvelope.model_validate(
            {
                "summary": "Useful resources",
                "changes": [
                    {
                        "path": "/workspace/src/App.tsx",
                        "operation": "replace_text",
                        "match": "<main>Old</main>",
                        "replacement": "<main>Resources</main>",
                    },
                    {
                        "path": "/workspace/src/App.tsx",
                        "operation": "replace_text",
                        "match": "<main>Resources</main>",
                        "replacement": "<section>Useful resources</section>",
                    },
                ],
            }
        ),
        base_files={
            "/surface.json": (
                '{"format":"aloy-react-surface","entrypoint":"/src/App.tsx",'
                '"sdk_version":"1"}'
            ),
            "/src/App.tsx": ("export default function App(){return <main>Old</main>}"),
        },
        primary_jobs=["See resources"],
    )

    files = {item.source_path: item.content for item in candidate.files}
    assert files["/src/App.tsx"] == (
        "export default function App(){return <section>Useful resources</section>}"
    )


async def test_redundant_edit_is_ignored_when_transaction_has_a_real_change():
    manifest = (
        '{"format":"aloy-react-surface","entrypoint":"/src/App.tsx",'
        '"sdk_version":"1"}'
    )
    candidate = materialize_surface_candidate_edit(
        SurfaceCandidateEditEnvelope.model_validate(
            {
                "summary": "Useful resources",
                "changes": [
                    {
                        "path": "/workspace/surface.json",
                        "operation": "write",
                        "content": manifest,
                    },
                    {
                        "path": "/workspace/src/App.tsx",
                        "operation": "replace_text",
                        "match": "<main>Old</main>",
                        "replacement": "<main>Resources</main>",
                    },
                ],
            }
        ),
        base_files={
            "/surface.json": manifest,
            "/src/App.tsx": "export default function App(){return <main>Old</main>}",
        },
        primary_jobs=["See resources"],
    )

    files = {item.source_path: item.content for item in candidate.files}
    assert files["/surface.json"] == manifest
    assert files["/src/App.tsx"].endswith("<main>Resources</main>}")


async def test_entirely_redundant_edit_transaction_is_rejected():
    manifest = (
        '{"format":"aloy-react-surface","entrypoint":"/src/App.tsx",'
        '"sdk_version":"1"}'
    )

    with pytest.raises(ValueError, match="transaction does not change source"):
        materialize_surface_candidate_edit(
            SurfaceCandidateEditEnvelope.model_validate(
                {
                    "summary": "No source change",
                    "changes": [
                        {
                            "path": "/workspace/surface.json",
                            "operation": "write",
                            "content": manifest,
                        }
                    ],
                }
            ),
            base_files={
                "/surface.json": manifest,
                "/src/App.tsx": (
                    "export default function App(){return <main>Old</main>}"
                ),
            },
            primary_jobs=["See resources"],
        )


async def test_ordered_edit_failure_leaves_frozen_source_unchanged():
    base_files = {
        "/surface.json": (
            '{"format":"aloy-react-surface","entrypoint":"/src/App.tsx",'
            '"sdk_version":"1"}'
        ),
        "/src/App.tsx": "export default function App(){return <main>Old</main>}",
    }

    with pytest.raises(ValueError, match="found 0 occurrences"):
        materialize_surface_candidate_edit(
            SurfaceCandidateEditEnvelope.model_validate(
                {
                    "summary": "Broken transaction",
                    "changes": [
                        {
                            "path": "/workspace/src/App.tsx",
                            "operation": "replace_text",
                            "match": "<main>Old</main>",
                            "replacement": "<main>Resources</main>",
                        },
                        {
                            "path": "/workspace/src/App.tsx",
                            "operation": "replace_text",
                            "match": "<footer>Missing</footer>",
                            "replacement": "<footer>Ready</footer>",
                        },
                    ],
                }
            ),
            base_files=base_files,
            primary_jobs=["See resources"],
        )

    assert base_files["/src/App.tsx"].endswith("<main>Old</main>}")


async def test_exact_text_edit_fails_closed_when_match_is_ambiguous():
    with pytest.raises(ValueError, match="found 2 occurrences"):
        materialize_surface_candidate_edit(
            SurfaceCandidateEditEnvelope.model_validate(
                {
                    "summary": "Ambiguous repair",
                    "changes": [
                        {
                            "path": "/workspace/src/App.tsx",
                            "operation": "replace_text",
                            "match": "Resources",
                            "replacement": "Files",
                        }
                    ],
                }
            ),
            base_files={
                "/surface.json": (
                    '{"format":"aloy-react-surface",'
                    '"entrypoint":"/src/App.tsx","sdk_version":"1"}'
                ),
                "/src/App.tsx": "const title='Resources'; const tab='Resources';",
            },
            primary_jobs=["See resources"],
        )


async def test_host_binds_model_job_proofs_to_frozen_contract():
    candidate = SurfaceCandidate.model_validate(
        {
            "summary": "Resources",
            "primary_jobs": ["model metadata is ignored"],
            "files": [
                {
                    "path": "/workspace/surface.json",
                    "content": (
                        '{"format":"aloy-react-surface","entrypoint":"/src/App.tsx",'
                        '"sdk_version":"1","capabilities":[],"intents":{},'
                        '"primary_jobs":[{"id":"job_0000000000000000",'
                        '"description":"Open a resource","steps":[],'
                        '"assertions":[{"kind":"visible","role":"heading",'
                        '"name":"Resources"}]}],"widgets":[]}'
                    ),
                },
                {
                    "path": "/workspace/src/App.tsx",
                    "content": "export default function App(){return <h1>Resources</h1>}",
                },
            ],
        }
    )

    rebound = bind_surface_manifest_primary_jobs(
        candidate,
        required_primary_jobs=[
            {"id": "job_1234567890abcdef", "description": "Open a resource"}
        ],
    )

    assert rebound.primary_jobs == ["Open a resource"]
    manifest = next(
        item.content for item in rebound.files if item.source_path == "/surface.json"
    )
    assert '"id": "job_1234567890abcdef"' in manifest
    assert "job_0000000000000000" not in manifest


async def test_host_pipeline_rejects_source_that_changed_after_context_freeze():
    with pytest.raises(SurfaceConflictError, match="context was frozen"):
        await SurfaceHostPipeline(
            run_id="run-stale-base",
            authoring_handler=FakeAuthoring(),
            build_handler=FakeBuilds(),
            expected_base_revision_id="different-revision",
        ).execute(_candidate(), submission=1)


class FakeAuthoring:
    def __init__(self) -> None:
        self.write_params = None

    async def read(self):
        return {
            "expected_revision": "revision-old",
            "draft": {
                "files": {
                    "/src/App.tsx": "old",
                    "/src/obsolete.css": "remove me",
                }
            },
        }

    async def write(self, params):
        self.write_params = params
        return {
            "draft": {"id": "revision-new"},
            "project": {
                "published_revision_id": "revision-live",
                "published_build_id": "build-live",
            },
        }


class FakeBuilds:
    def __init__(
        self,
        *,
        status: str = "succeeded",
        diagnostic_code: str = "typescript_error",
        preview_diagnostics: list[dict] | None = None,
    ) -> None:
        self.status = status
        self.diagnostic_code = diagnostic_code
        self.preview_diagnostics = preview_diagnostics
        self.calls: list[str] = []

    async def build(self, params):
        self.calls.append("build")
        return {
            "id": "build-new",
            "status": self.status,
            "diagnostics": (
                []
                if self.status == "succeeded"
                else [
                    {
                        "code": self.diagnostic_code,
                        "message": "App.tsx does not type check",
                        "path": "/src/App.tsx",
                        "line": 4,
                    }
                ]
            ),
        }

    async def preview(self, params):
        self.calls.append("preview")
        return {
            "preview_ready": self.preview_diagnostics is None,
            "diagnostics": self.preview_diagnostics or [],
        }

    async def publish(self, params):
        self.calls.append("publish")
        assert params.expected_published_revision_id == "revision-live"
        assert params.expected_published_build_id == "build-live"
        return {"id": "publication-new", "build_id": params.build_id}


async def test_host_pipeline_replaces_source_and_owns_every_lifecycle_stage():
    authoring = FakeAuthoring()
    builds = FakeBuilds()
    stages: list[str] = []

    async def observe(stage: str) -> None:
        stages.append(stage)

    result = await SurfaceHostPipeline(
        run_id="run-1",
        authoring_handler=authoring,
        build_handler=builds,
        stage_observer=observe,
    ).execute(_candidate(), submission=1)

    assert result.status == "published"
    assert result.revision_id == "revision-new"
    assert result.build_id == "build-new"
    assert result.publication == {
        "id": "publication-new",
        "build_id": "build-new",
    }
    assert builds.calls == ["build", "preview", "publish"]
    assert stages == [
        "validating_candidate",
        "building_bundle",
        "inspecting_preview",
        "publishing_surface",
    ]
    assert authoring.write_params.expected_revision == "revision-old"
    patches = {
        (patch.path, patch.operation): patch.content
        for patch in authoring.write_params.patches
    }
    assert patches[("/workspace/src/obsolete.css", "delete")] is None
    assert "/workspace/src/App.tsx" in {
        path for path, operation in patches if operation == "write"
    }
    assert set(result.timings_ms) == {"persist", "build", "preview", "publish"}


async def test_persisted_revision_uses_the_same_build_preview_publication_path():
    builds = FakeBuilds()
    stages: list[str] = []

    async def observe(stage: str) -> None:
        stages.append(stage)

    result = await SurfaceRevisionHostPipeline(
        run_id="run-existing-source",
        build_handler=builds,
        stage_observer=observe,
    ).execute(
        revision_id="revision-reviewed",
        source_fingerprint="a" * 64,
        expected_published_revision_id="revision-live",
        expected_published_build_id="build-live",
        attempt=1,
    )

    assert result.status == "published"
    assert result.revision_id == "revision-reviewed"
    assert builds.calls == ["build", "preview", "publish"]
    assert stages == [
        "building_bundle",
        "inspecting_preview",
        "publishing_surface",
    ]


async def test_host_pipeline_rejects_candidate_that_redefines_frozen_jobs():
    authoring = FakeAuthoring()
    builds = FakeBuilds()
    result = await SurfaceHostPipeline(
        run_id="run-primary-job-mismatch",
        authoring_handler=authoring,
        build_handler=builds,
        required_primary_jobs=[
            {"id": "job_0123456789abcdef", "description": "See the week"}
        ],
    ).execute(_candidate(), submission=1)

    assert result.status == "repair_required"
    assert authoring.write_params is None
    assert builds.calls == []
    assert {item["code"] for item in result.diagnostics} == {
        "primary_job_contract_mismatch",
        "primary_job_manifest_mismatch",
    }


async def test_host_pipeline_accepts_exact_host_frozen_primary_job_contract():
    value = _candidate().model_dump(mode="python")
    value["primary_jobs"] = ["See the week"]
    value["files"][0]["content"] = (
        '{"format":"aloy-react-surface","entrypoint":"/src/App.tsx",'
        '"sdk_version":"1","capabilities":[],"intents":{},'
        '"interaction_checks":[],"primary_jobs":[{'
        '"id":"job_0123456789abcdef","description":"See the week",'
        '"steps":[],"assertions":[{"kind":"visible","role":"heading",'
        '"name":"University"}]}],"widgets":[]}'
    )
    candidate = SurfaceCandidate.model_validate(value)
    authoring = FakeAuthoring()
    builds = FakeBuilds()
    result = await SurfaceHostPipeline(
        run_id="run-primary-job-exact",
        authoring_handler=authoring,
        build_handler=builds,
        required_primary_jobs=[
            {"id": "job_0123456789abcdef", "description": "See the week"}
        ],
    ).execute(candidate, submission=1)

    assert result.status == "published"
    assert authoring.write_params is not None
    assert builds.calls == ["build", "preview", "publish"]


async def test_host_pipeline_returns_repair_diagnostics_without_publication():
    builds = FakeBuilds(status="failed")
    result = await SurfaceHostPipeline(
        run_id="run-2",
        authoring_handler=FakeAuthoring(),
        build_handler=builds,
    ).execute(_candidate(), submission=1)

    assert result.status == "repair_required"
    assert result.publication is None
    assert builds.calls == ["build"]
    assert result.diagnostics[0] == {
        "stage": "build",
        "code": "typescript_error",
        "severity": "error",
        "message": "App.tsx does not type check",
        "path": "/src/App.tsx",
        "line": 4,
    }


async def test_host_pipeline_separates_infrastructure_failure_from_model_repair():
    builds = FakeBuilds(status="failed", diagnostic_code="bundle_storage_failed")
    result = await SurfaceHostPipeline(
        run_id="run-storage",
        authoring_handler=FakeAuthoring(),
        build_handler=builds,
    ).execute(_candidate(), submission=1)

    assert result.status == "host_failed"
    assert result.revision_id == "revision-new"
    assert result.build_id == "build-new"
    assert result.diagnostics[0]["code"] == "bundle_storage_failed"
    assert builds.calls == ["build"]


async def test_runtime_exception_requests_model_repair_before_publication():
    builds = FakeBuilds(
        preview_diagnostics=[
            {
                "code": "runtime_exception",
                "message": "Cannot read properties of undefined",
            }
        ]
    )
    result = await SurfaceHostPipeline(
        run_id="run-runtime-repair",
        authoring_handler=FakeAuthoring(),
        build_handler=builds,
    ).execute(_candidate(), submission=1)

    assert result.status == "repair_required"
    assert result.diagnostics[0]["code"] == "runtime_exception"
    assert builds.calls == ["build", "preview"]


async def test_runtime_inspector_failure_is_not_sent_to_model():
    builds = FakeBuilds(
        preview_diagnostics=[
            {
                "code": "runtime_inspector_unavailable",
                "message": "Headless browser unavailable",
            }
        ]
    )
    result = await SurfaceHostPipeline(
        run_id="run-runtime-host-failure",
        authoring_handler=FakeAuthoring(),
        build_handler=builds,
    ).execute(_candidate(), submission=1)

    assert result.status == "host_failed"
    assert builds.calls == ["build", "preview"]


async def test_host_pipeline_turns_source_validation_into_repair_diagnostics():
    class RejectingAuthoring(FakeAuthoring):
        async def write(self, params):
            raise SurfaceAuthoringError("Surface manifest is missing an entrypoint")

    builds = FakeBuilds()
    result = await SurfaceHostPipeline(
        run_id="run-validation",
        authoring_handler=RejectingAuthoring(),
        build_handler=builds,
    ).execute(_candidate(), submission=1)

    assert result.status == "repair_required"
    assert result.revision_id is None
    assert builds.calls == []
    assert result.diagnostics[0]["stage"] == "validation"
    assert result.diagnostics[0]["code"] == "source_rejected"
    assert "entrypoint" in result.diagnostics[0]["message"]


def test_candidate_is_complete_unique_and_workspace_scoped():
    with pytest.raises(ValidationError, match="surface.json"):
        SurfaceCandidate.model_validate(
            {
                "summary": "Incomplete",
                "primary_jobs": ["See the week"],
                "files": [
                    {
                        "path": "/workspace/src/App.tsx",
                        "content": "export default function App() { return null }",
                    }
                ],
            }
        )

    with pytest.raises(ValidationError, match="unique"):
        value = _candidate().model_dump(mode="json")
        value["files"].append(value["files"][0])
        SurfaceCandidate.model_validate(value)

    with pytest.raises(ValidationError, match="workspace"):
        value = _candidate().model_dump(mode="json")
        value["files"][0]["path"] = "/tmp/surface.json"
        SurfaceCandidate.model_validate(value)


def test_candidate_normalizes_safe_project_paths_and_bounded_summary():
    value = _candidate().model_dump(mode="json")
    value["summary"] = "  " + ("Useful career workspace.  " * 100) + "  "
    value["files"][0]["path"] = "/surface.json"
    value["files"][1]["path"] = "src/App.tsx"

    candidate = SurfaceCandidate.model_validate(value)

    assert len(candidate.summary) == 1000
    assert "  " not in candidate.summary
    assert [item.path for item in candidate.files] == [
        "/workspace/surface.json",
        "/workspace/src/App.tsx",
    ]
