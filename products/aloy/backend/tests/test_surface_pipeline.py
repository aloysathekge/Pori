from __future__ import annotations

import pytest
from pydantic import ValidationError

from aloy_backend.surface_authoring import SurfaceAuthoringError
from aloy_backend.surface_pipeline import (
    SurfaceCandidate,
    SurfaceHostPipeline,
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
