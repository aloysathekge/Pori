from __future__ import annotations

from types import SimpleNamespace

from aloy_backend.surface_manifest import SurfaceManifest
from aloy_backend.surface_quality import (
    REQUIRED_SURFACE_VIEWPORTS,
    SURFACE_QUALITY_RECEIPT_KEY,
    create_surface_quality_receipt,
    surface_quality_receipt_error,
)


def _inspection_evidence() -> dict:
    required = [str(item["id"]) for item in REQUIRED_SURFACE_VIEWPORTS]
    return {
        "viewport_matrix": {
            "policy_version": "aloy-surface-viewports@1",
            "required": required,
            "passed": True,
            "viewports": [
                {
                    "id": viewport_id,
                    "capture": {"sha256": f"capture-{viewport_id}"},
                    "accessibility": {
                        "main_landmarks": 1,
                        "unnamed_controls": 0,
                        "images_missing_alt": 0,
                        "keyboard_unreachable": 0,
                        "duplicate_ids": [],
                    },
                }
                for viewport_id in required
            ],
        }
    }


def _build(receipt: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id="build-1",
        revision_id="revision-1",
        source_checksum="source-sha",
        bundle_sha256="bundle-sha",
        resource_metrics=(
            {SURFACE_QUALITY_RECEIPT_KEY: receipt} if receipt is not None else {}
        ),
    )


def test_quality_receipt_is_bound_to_exact_source_and_bundle():
    receipt = create_surface_quality_receipt(
        build_id="build-1",
        revision_id="revision-1",
        source_checksum="source-sha",
        bundle_sha256="bundle-sha",
        validation_passed=True,
        manifest=SurfaceManifest(),
        runtime_proven=True,
        runtime_diagnostics=[],
        inspection_evidence=_inspection_evidence(),
    )
    build = _build(receipt)

    assert receipt["passed"] is True
    assert surface_quality_receipt_error(build) is None

    build.bundle_sha256 = "different-bundle"
    assert "different source or bundle" in (surface_quality_receipt_error(build) or "")


def test_quality_receipt_fails_closed_for_runtime_failure_and_tampering():
    receipt = create_surface_quality_receipt(
        build_id="build-1",
        revision_id="revision-1",
        source_checksum="source-sha",
        bundle_sha256="bundle-sha",
        validation_passed=True,
        manifest=SurfaceManifest(),
        runtime_proven=False,
        runtime_diagnostics=[{"code": "runtime_exception", "message": "render failed"}],
        inspection_evidence=_inspection_evidence(),
    )
    assert "did not pass" in (surface_quality_receipt_error(_build(receipt)) or "")

    receipt["passed"] = True
    assert "fingerprint is invalid" in (
        surface_quality_receipt_error(_build(receipt)) or ""
    )
