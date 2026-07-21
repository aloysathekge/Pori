"""Provider-neutral transport for trusted Surface runtime inspection.

Compilation and browser inspection are separate authority boundaries.  A remote
inspector may execute Chromium, but it never receives ObjectStore credentials
or publication authority: it returns bounded evidence bytes to Aloy's host,
which verifies and retains them.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Any, Protocol

from .surface_manifest import SurfaceManifest
from .surface_runtime import SurfaceRuntimeDocument
from .surface_runtime_inspection import inspect_surface_runtime

SURFACE_INSPECTION_TRANSPORT_VERSION = "1"
MAX_SURFACE_INSPECTION_ARTIFACT_BYTES = 4 * 1024 * 1024
MAX_SURFACE_INSPECTION_ARTIFACTS = 8
ALLOWED_SURFACE_INSPECTION_ARTIFACTS = {
    ("viewport_capture", "image/png"),
}


@dataclass(frozen=True)
class SurfaceInspectionBinding:
    """Exact immutable inputs an inspector is permitted to examine."""

    build_id: str
    revision_id: str
    source_checksum: str
    bundle_sha256: str
    nonce: str


@dataclass(frozen=True)
class SurfaceInspectionRequest:
    binding: SurfaceInspectionBinding
    document: SurfaceRuntimeDocument
    runtime_context: dict[str, Any]
    manifest: SurfaceManifest


@dataclass(frozen=True)
class SurfaceInspectionArtifact:
    """Untrusted binary evidence returned to the host for re-hashing."""

    kind: str
    name: str
    content_type: str
    data: bytes
    sha256: str


@dataclass(frozen=True)
class SurfaceInspectionResult:
    binding: SurfaceInspectionBinding
    diagnostics: list[dict[str, Any]]
    evidence: dict[str, Any]
    artifacts: list[SurfaceInspectionArtifact] = field(default_factory=list)
    transport: dict[str, Any] = field(default_factory=dict)


class SurfaceInspectionTransport(Protocol):
    """A local or remote inspector with no storage or publication authority."""

    transport_id: str

    def inspect(self, request: SurfaceInspectionRequest) -> SurfaceInspectionResult: ...


def new_surface_inspection_binding(
    *,
    build_id: str,
    revision_id: str,
    source_checksum: str,
    bundle_sha256: str,
) -> SurfaceInspectionBinding:
    return SurfaceInspectionBinding(
        build_id=build_id,
        revision_id=revision_id,
        source_checksum=source_checksum,
        bundle_sha256=bundle_sha256,
        nonce=secrets.token_urlsafe(24),
    )


class LocalSurfaceInspectionTransport:
    """Adapter for the existing trusted local Chromium inspector."""

    transport_id = "local-chromium"

    def inspect(self, request: SurfaceInspectionRequest) -> SurfaceInspectionResult:
        evidence: dict[str, Any] = {}
        diagnostics = inspect_surface_runtime(
            request.document,
            request.runtime_context,
            manifest=request.manifest,
            evidence_sink=evidence,
        )
        artifacts: list[SurfaceInspectionArtifact] = []
        for capture in list(evidence.pop("_capture_blobs", []) or []):
            if not isinstance(capture, dict):
                continue
            data = capture.get("data")
            if not isinstance(data, bytes):
                continue
            artifacts.append(
                SurfaceInspectionArtifact(
                    kind="viewport_capture",
                    name=str(capture.get("name") or "viewport.png"),
                    content_type=str(capture.get("content_type") or "image/png"),
                    data=data,
                    sha256=str(capture.get("sha256") or ""),
                )
            )
        return SurfaceInspectionResult(
            binding=request.binding,
            diagnostics=diagnostics,
            evidence=evidence,
            artifacts=artifacts,
            transport={
                "id": self.transport_id,
                "version": SURFACE_INSPECTION_TRANSPORT_VERSION,
            },
        )


def validate_surface_inspection_result(
    result: SurfaceInspectionResult,
    *,
    expected: SurfaceInspectionBinding,
) -> list[dict[str, Any]]:
    """Fail closed before any remote evidence reaches durable storage."""

    diagnostics: list[dict[str, Any]] = []
    if result.binding != expected:
        diagnostics.append(
            _diagnostic(
                "inspection_binding_mismatch",
                "The inspector result was not bound to this exact Surface build",
            )
        )
    if len(result.artifacts) > MAX_SURFACE_INSPECTION_ARTIFACTS:
        diagnostics.append(
            _diagnostic(
                "inspection_artifact_limit",
                "The inspector returned too many evidence artifacts",
            )
        )
    seen: set[tuple[str, str]] = set()
    retained_checksums: set[str] = set()
    for artifact in result.artifacts:
        identity = (artifact.kind, artifact.name)
        if identity in seen:
            diagnostics.append(
                _diagnostic(
                    "inspection_artifact_duplicate",
                    "The inspector returned duplicate evidence artifacts",
                )
            )
            continue
        seen.add(identity)
        if (
            artifact.kind,
            artifact.content_type,
        ) not in ALLOWED_SURFACE_INSPECTION_ARTIFACTS:
            diagnostics.append(
                _diagnostic(
                    "inspection_artifact_unsupported",
                    "The inspector returned an unsupported evidence artifact type",
                )
            )
            continue
        if (
            not artifact.data
            or len(artifact.data) > MAX_SURFACE_INSPECTION_ARTIFACT_BYTES
        ):
            diagnostics.append(
                _diagnostic(
                    "inspection_artifact_invalid",
                    "An inspector evidence artifact was empty or exceeded its size limit",
                )
            )
            continue
        actual = hashlib.sha256(artifact.data).hexdigest()
        if artifact.sha256 != actual:
            diagnostics.append(
                _diagnostic(
                    "inspection_artifact_checksum_mismatch",
                    "An inspector evidence artifact checksum did not match its bytes",
                )
            )
            continue
        retained_checksums.add(actual)
    viewport_matrix = dict(result.evidence.get("viewport_matrix") or {})
    for viewport in list(viewport_matrix.get("viewports") or []):
        if not isinstance(viewport, dict):
            continue
        capture = dict(viewport.get("capture") or {})
        checksum = str(capture.get("sha256") or "")
        if checksum and checksum not in retained_checksums:
            diagnostics.append(
                _diagnostic(
                    "inspection_capture_missing",
                    "The inspector claimed a viewport capture that was not returned",
                )
            )
    return diagnostics


def _diagnostic(code: str, message: str) -> dict[str, Any]:
    return {
        "stage": "inspection_transport",
        "code": code,
        "severity": "error",
        "message": message,
        "path": None,
        "line": None,
    }


__all__ = [
    "LocalSurfaceInspectionTransport",
    "ALLOWED_SURFACE_INSPECTION_ARTIFACTS",
    "MAX_SURFACE_INSPECTION_ARTIFACT_BYTES",
    "MAX_SURFACE_INSPECTION_ARTIFACTS",
    "SURFACE_INSPECTION_TRANSPORT_VERSION",
    "SurfaceInspectionArtifact",
    "SurfaceInspectionBinding",
    "SurfaceInspectionRequest",
    "SurfaceInspectionResult",
    "SurfaceInspectionTransport",
    "new_surface_inspection_binding",
    "validate_surface_inspection_result",
]
