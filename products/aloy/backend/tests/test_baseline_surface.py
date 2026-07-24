"""The bundled baseline Surface must pass the full trusted gate as-is.

S1 of ``docs/aloy-baseline-surface-spec.md``: prove the template compiles,
validates, and passes browser inspection with primary-job proofs against both
a fresh (empty) Event and a populated one — before any delivery wiring exists.
"""

from __future__ import annotations

import json
import shutil

import pytest

from aloy_backend.baseline_surface import (
    baseline_surface_files,
    baseline_surface_fingerprint,
)
from aloy_backend.surface_build_runner import (
    LocalDevelopmentSurfaceBuildRunner,
    validate_surface_source,
)
from aloy_backend.surface_manifest import SurfaceManifest, parse_surface_manifest
from aloy_backend.surface_runtime import build_surface_runtime_document
from aloy_backend.surface_runtime_inspection import inspect_surface_runtime

pytestmark = pytest.mark.asyncio


def _context(*, tasks: list[dict], files: list[dict]) -> dict:
    return {
        "protocol_version": "1",
        "sdk_version": "1",
        "event_id": "event-baseline",
        "project_id": "project-baseline",
        "build_id": "build-baseline",
        "code_revision_id": "revision-baseline",
        "data_revision": 0,
        "capabilities": ["tasks", "files", "ask_aloy"],
        "widgets": [],
        "data": {
            "event": {
                "title": "Plan the launch",
                "summary": "Everything for the spring product launch.",
            },
            "tasks": tasks,
            "files": files,
            "interactions": [],
        },
    }


def test_baseline_source_passes_static_validation() -> None:
    files = baseline_surface_files()
    manifest_value = json.loads(files["/surface.json"])
    manifest = parse_surface_manifest(files)
    assert isinstance(manifest, SurfaceManifest)
    assert validate_surface_source(files, manifest_value) == []
    assert len(baseline_surface_fingerprint()) == 64


async def test_baseline_passes_full_gate_on_a_fresh_empty_event() -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js is not installed")
    files = baseline_surface_files()
    manifest = parse_surface_manifest(files)
    result = await LocalDevelopmentSurfaceBuildRunner().build(
        build_id="baseline-empty-event",
        files=files,
        manifest=manifest.model_dump(mode="json", by_alias=True),
    )
    if result.status == "blocked":
        pytest.skip("Pinned Aloy app dependencies are not installed")
    assert result.status == "succeeded", result.diagnostics
    assert result.bundle is not None
    evidence: dict = {}
    diagnostics = inspect_surface_runtime(
        build_surface_runtime_document(result.bundle),
        _context(tasks=[], files=[]),
        manifest=manifest,
        evidence_sink=evidence,
    )
    assert diagnostics == []
    assert evidence["primary_jobs"]["passed"] is True
    assert evidence["primary_jobs"]["required"] == [
        "job_ba5e11e5a1b2c3d4",
        "job_ba5e11e5a1b2c3d5",
        "job_ba5e11e5a1b2c3d6",
    ]


async def test_baseline_passes_full_gate_with_populated_event_data() -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js is not installed")
    files = baseline_surface_files()
    manifest = parse_surface_manifest(files)
    result = await LocalDevelopmentSurfaceBuildRunner().build(
        build_id="baseline-populated-event",
        files=files,
        manifest=manifest.model_dump(mode="json", by_alias=True),
    )
    if result.status == "blocked":
        pytest.skip("Pinned Aloy app dependencies are not installed")
    assert result.status == "succeeded", result.diagnostics
    assert result.bundle is not None
    diagnostics = inspect_surface_runtime(
        build_surface_runtime_document(result.bundle),
        _context(
            tasks=[
                {"id": "task-1", "title": "Book the venue", "status": "in_progress"},
                {
                    "id": "task-2",
                    "title": "Draft the invite list",
                    "status": "completed",
                },
            ],
            files=[
                {
                    "id": "file-1",
                    "name": "launch-plan.pdf",
                    "kind": "upload",
                    "mime_type": "application/pdf",
                    "size_bytes": 2048,
                }
            ],
        ),
        manifest=manifest,
    )
    assert diagnostics == []


def test_declared_capability_without_wired_region_fails_static_validation() -> None:
    files = dict(baseline_surface_files())
    manifest_value = json.loads(files["/surface.json"])
    manifest_value["capabilities"] = sorted(
        [*manifest_value["capabilities"], "data:career"]
    )
    files["/surface.json"] = json.dumps(manifest_value)
    diagnostics = validate_surface_source(files, manifest_value)
    assert [item["code"] for item in diagnostics] == ["resource_state_unwired"]
    assert "data:career" in diagnostics[0]["message"]

    files["/src/views/Career.tsx"] = (
        "import { useSurfaceResourceState } from '@aloy/surface';\n"
        "export function CareerView() {\n"
        "  const state = useSurfaceResourceState('data:career');\n"
        "  return <section {...state.feedbackProps}>Career</section>;\n"
        "}\n"
    )
    assert validate_surface_source(files, manifest_value) == []
