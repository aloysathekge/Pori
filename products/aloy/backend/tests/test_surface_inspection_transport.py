"""Trust-boundary tests for local and future remote Surface inspectors."""

from __future__ import annotations

import base64
import hashlib
import json

from aloy_backend.surface_inspection_transport import (
    REMOTE_SURFACE_INSPECTION_COMMAND,
    SandboxSurfaceInspectionTransport,
    SurfaceInspectionArtifact,
    SurfaceInspectionRequest,
    SurfaceInspectionResult,
    new_surface_inspection_binding,
    validate_surface_inspection_result,
)
from aloy_backend.surface_manifest import SurfaceManifest
from aloy_backend.surface_runtime import SurfaceRuntimeDocument


def _binding():
    return new_surface_inspection_binding(
        build_id="sbuild_test",
        revision_id="srev_test",
        source_checksum="source-sha",
        bundle_sha256="bundle-sha",
    )


def _result(*, binding, artifact: SurfaceInspectionArtifact):
    return SurfaceInspectionResult(
        binding=binding,
        diagnostics=[],
        evidence={},
        artifacts=[artifact],
        transport={"id": "remote-browser", "version": "1"},
    )


def test_inspection_transport_accepts_exact_bound_png_evidence():
    binding = _binding()
    data = b"trusted-png-bytes"
    result = _result(
        binding=binding,
        artifact=SurfaceInspectionArtifact(
            kind="viewport_capture",
            name="wide.png",
            content_type="image/png",
            data=data,
            sha256=hashlib.sha256(data).hexdigest(),
        ),
    )

    assert validate_surface_inspection_result(result, expected=binding) == []


def test_inspection_transport_rejects_misbound_or_unsupported_evidence():
    binding = _binding()
    other_binding = _binding()
    data = b"not-a-png"
    result = _result(
        binding=other_binding,
        artifact=SurfaceInspectionArtifact(
            kind="arbitrary_file",
            name="secret.txt",
            content_type="text/plain",
            data=data,
            sha256="incorrect",
        ),
    )

    assert {
        item["code"]
        for item in validate_surface_inspection_result(result, expected=binding)
    } == {
        "inspection_binding_mismatch",
        "inspection_artifact_unsupported",
    }


class FakeInspectionSandbox:
    def __init__(self, result: dict):
        self.result = result
        self.writes: dict[str, str] = {}
        self.commands: list[str] = []

    def write_file(self, path, content, append=False):
        assert append is False
        self.writes[path] = content

    def execute_command(self, command, cwd=None):
        assert cwd is None
        self.commands.append(command)
        return "inspection complete"

    def read_file(self, path):
        assert path == "/inspection/result.json"
        return json.dumps(self.result)


class FakeInspectionProvider:
    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.acquired: list[str] = []
        self.released: list[str] = []

    def acquire(self, key):
        self.acquired.append(key)
        return "remote-sandbox"

    def get(self, sandbox_id):
        assert sandbox_id == "remote-sandbox"
        return self.sandbox

    def release(self, sandbox_id):
        self.released.append(sandbox_id)


def test_sandbox_transport_uses_fixed_toolchain_and_returns_bounded_result():
    binding = _binding()
    data = b"remote-png"
    result = {
        "binding": {
            "build_id": binding.build_id,
            "revision_id": binding.revision_id,
            "source_checksum": binding.source_checksum,
            "bundle_sha256": binding.bundle_sha256,
            "nonce": binding.nonce,
        },
        "diagnostics": [],
        "evidence": {},
        "artifacts": [
            {
                "kind": "viewport_capture",
                "name": "wide.png",
                "content_type": "image/png",
                "data_base64": base64.b64encode(data).decode("ascii"),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ],
        "transport": {"id": "e2b-chromium", "version": "1"},
    }
    sandbox = FakeInspectionSandbox(result)
    provider = FakeInspectionProvider(sandbox)
    transport = SandboxSurfaceInspectionTransport(provider)  # type: ignore[arg-type]

    inspected = transport.inspect(
        SurfaceInspectionRequest(
            binding=binding,
            document=SurfaceRuntimeDocument(
                html="<main>Surface</main>",
                content_security_policy="default-src 'none'",
            ),
            runtime_context={"event_id": "evt_test", "capabilities": []},
            manifest=SurfaceManifest(),
        )
    )

    request = json.loads(sandbox.writes["/inspection/request.json"])
    assert sandbox.commands == [REMOTE_SURFACE_INSPECTION_COMMAND]
    assert request["binding"]["nonce"] == binding.nonce
    assert request["document"]["html"] == "<main>Surface</main>"
    assert inspected.transport == {"id": "e2b-chromium", "version": "1"}
    assert validate_surface_inspection_result(inspected, expected=binding) == []
    assert provider.released == ["remote-sandbox"]
