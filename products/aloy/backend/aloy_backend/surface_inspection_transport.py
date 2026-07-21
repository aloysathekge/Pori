"""Provider-neutral transport for trusted Surface runtime inspection.

Compilation and browser inspection are separate authority boundaries.  A remote
inspector may execute Chromium, but it never receives ObjectStore credentials
or publication authority: it returns bounded evidence bytes to Aloy's host,
which verifies and retains them.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import secrets
from dataclasses import dataclass, field
from typing import Any, Protocol

from pori import LocalSandboxProvider, SandboxProvider, get_sandbox_provider

from .surface_manifest import SurfaceManifest
from .surface_runtime import SurfaceRuntimeDocument
from .surface_runtime_inspection import inspect_surface_runtime

SURFACE_INSPECTION_TRANSPORT_VERSION = "1"
MAX_SURFACE_INSPECTION_ARTIFACT_BYTES = 4 * 1024 * 1024
MAX_SURFACE_INSPECTION_ARTIFACTS = 8
ALLOWED_SURFACE_INSPECTION_ARTIFACTS = {
    ("viewport_capture", "image/png"),
}
REMOTE_SURFACE_INSPECTION_COMMAND = (
    "timeout 60s /opt/aloy-surface-toolchain/bin/inspect-surface "
    "--request /inspection/request.json --output /inspection/result.json"
)


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


class SandboxSurfaceInspectionTransport:
    """Execute the fixed browser inspector in a secretless remote sandbox.

    The sandbox only receives a host-constructed document, capability-scoped
    runtime context, and manifest. It cannot address Aloy's database or object
    store; the host parses, binds, validates, hashes, and stores its output.
    """

    transport_id = "isolated-surface-browser"

    def __init__(self, provider: SandboxProvider):
        if isinstance(provider, LocalSandboxProvider):
            raise ValueError("Local host subprocesses cannot inspect Surfaces")
        self._provider = provider

    def inspect(self, request: SurfaceInspectionRequest) -> SurfaceInspectionResult:
        sandbox_id = self._provider.acquire(
            f"surface-inspection:{request.binding.build_id}:{request.binding.nonce}"
        )
        sandbox = self._provider.get(sandbox_id)
        if sandbox is None:
            return _unavailable_result(
                request,
                "The isolated browser inspection sandbox could not be acquired",
            )
        try:
            sandbox.write_file(
                "/inspection/request.json",
                json.dumps(
                    {
                        "transport_version": SURFACE_INSPECTION_TRANSPORT_VERSION,
                        "binding": _binding_payload(request.binding),
                        "document": {
                            "html": request.document.html,
                            "content_security_policy": request.document.content_security_policy,
                        },
                        "runtime_context": request.runtime_context,
                        "manifest": request.manifest.model_dump(mode="json"),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            output = sandbox.execute_command(REMOTE_SURFACE_INSPECTION_COMMAND)
            if output.startswith("Error:") or " exit_code=" in output:
                return _unavailable_result(
                    request,
                    "The isolated browser inspector reported an execution failure",
                )
            return _parse_sandbox_result(
                request,
                sandbox.read_file("/inspection/result.json"),
            )
        except Exception as exc:
            return _unavailable_result(
                request,
                f"The isolated browser inspector did not satisfy its contract: {exc}",
            )
        finally:
            self._provider.release(sandbox_id)


class UnavailableSurfaceInspectionTransport:
    """Fail closed when no non-local inspection provider is configured."""

    transport_id = "unavailable"

    def inspect(self, request: SurfaceInspectionRequest) -> SurfaceInspectionResult:
        return _unavailable_result(
            request,
            "No isolated browser inspection provider is configured",
        )


def configured_surface_inspection_transport() -> SurfaceInspectionTransport:
    """Select local developer inspection or the production isolated transport."""

    from .config import settings

    if settings.surface_build_backend == "local_dev":
        return LocalSurfaceInspectionTransport()
    provider = get_sandbox_provider()
    if provider is None or isinstance(provider, LocalSandboxProvider):
        return UnavailableSurfaceInspectionTransport()
    return SandboxSurfaceInspectionTransport(provider)


def _binding_payload(binding: SurfaceInspectionBinding) -> dict[str, str]:
    return {
        "build_id": binding.build_id,
        "revision_id": binding.revision_id,
        "source_checksum": binding.source_checksum,
        "bundle_sha256": binding.bundle_sha256,
        "nonce": binding.nonce,
    }


def _binding_from_payload(value: Any) -> SurfaceInspectionBinding:
    if not isinstance(value, dict):
        raise ValueError("inspection result binding is invalid")
    required = ("build_id", "revision_id", "source_checksum", "bundle_sha256", "nonce")
    if any(not isinstance(value.get(key), str) for key in required):
        raise ValueError("inspection result binding is incomplete")
    return SurfaceInspectionBinding(**{key: value[key] for key in required})


def _parse_sandbox_result(
    request: SurfaceInspectionRequest,
    raw: str,
) -> SurfaceInspectionResult:
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("inspection result must be an object")
        diagnostics = value.get("diagnostics")
        evidence = value.get("evidence")
        artifacts_value = value.get("artifacts", [])
        transport = value.get("transport", {})
        if not isinstance(diagnostics, list) or not all(
            isinstance(item, dict) for item in diagnostics
        ):
            raise ValueError("inspection diagnostics are invalid")
        if not isinstance(evidence, dict) or not isinstance(artifacts_value, list):
            raise ValueError("inspection evidence is invalid")
        if not isinstance(transport, dict):
            raise ValueError("inspection transport metadata is invalid")
        artifacts: list[SurfaceInspectionArtifact] = []
        for item in artifacts_value:
            if not isinstance(item, dict):
                raise ValueError("inspection artifact is invalid")
            encoded = item.get("data_base64")
            if not isinstance(encoded, str):
                raise ValueError("inspection artifact data is invalid")
            artifacts.append(
                SurfaceInspectionArtifact(
                    kind=str(item.get("kind") or ""),
                    name=str(item.get("name") or ""),
                    content_type=str(item.get("content_type") or ""),
                    data=base64.b64decode(encoded, validate=True),
                    sha256=str(item.get("sha256") or ""),
                )
            )
        return SurfaceInspectionResult(
            binding=_binding_from_payload(value.get("binding")),
            diagnostics=[dict(item) for item in diagnostics],
            evidence=dict(evidence),
            artifacts=artifacts,
            transport={
                "id": str(
                    transport.get("id")
                    or SandboxSurfaceInspectionTransport.transport_id
                ),
                "version": str(
                    transport.get("version") or SURFACE_INSPECTION_TRANSPORT_VERSION
                ),
            },
        )
    except (TypeError, ValueError, binascii.Error) as exc:
        return _unavailable_result(
            request,
            f"The isolated browser returned an invalid inspection result: {exc}",
        )


def _unavailable_result(
    request: SurfaceInspectionRequest,
    message: str,
) -> SurfaceInspectionResult:
    return SurfaceInspectionResult(
        binding=request.binding,
        diagnostics=[
            {
                "stage": "inspection_transport",
                "code": "isolated_inspector_unavailable",
                "severity": "error",
                "message": message[:4000],
                "path": None,
                "line": None,
            }
        ],
        evidence={},
        transport={
            "id": UnavailableSurfaceInspectionTransport.transport_id,
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
    "ALLOWED_SURFACE_INSPECTION_ARTIFACTS",
    "MAX_SURFACE_INSPECTION_ARTIFACT_BYTES",
    "MAX_SURFACE_INSPECTION_ARTIFACTS",
    "REMOTE_SURFACE_INSPECTION_COMMAND",
    "SURFACE_INSPECTION_TRANSPORT_VERSION",
    "LocalSurfaceInspectionTransport",
    "SandboxSurfaceInspectionTransport",
    "SurfaceInspectionArtifact",
    "SurfaceInspectionBinding",
    "SurfaceInspectionRequest",
    "SurfaceInspectionResult",
    "SurfaceInspectionTransport",
    "UnavailableSurfaceInspectionTransport",
    "configured_surface_inspection_transport",
    "new_surface_inspection_binding",
    "validate_surface_inspection_result",
]
