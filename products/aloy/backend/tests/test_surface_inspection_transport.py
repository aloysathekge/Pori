"""Trust-boundary tests for local and future remote Surface inspectors."""

from __future__ import annotations

import hashlib

from aloy_backend.surface_inspection_transport import (
    SurfaceInspectionArtifact,
    SurfaceInspectionResult,
    new_surface_inspection_binding,
    validate_surface_inspection_result,
)


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
