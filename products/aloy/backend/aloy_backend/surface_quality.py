"""Host-owned quality receipts for exact Surface build publication.

The first R9 policy binds the deterministic source gate, executable runtime
inspection, and every manifest-declared interaction check to one immutable
build. Later R9 policies extend the same receipt with viewport, accessibility,
Critic, and primary-job evidence without moving publication authority into
generated code or a model.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from .surface_manifest import SurfaceManifest
from .surface_resource_states import REQUIRED_SURFACE_STATE_FIXTURES

SURFACE_QUALITY_POLICY_VERSION = "aloy-surface-quality@2"
SURFACE_QUALITY_RECEIPT_KEY = "surface_quality"

REQUIRED_SURFACE_VIEWPORTS: tuple[dict[str, Any], ...] = (
    {"id": "wide", "width": 1440, "height": 900, "compact": False},
    {"id": "split", "width": 640, "height": 900, "compact": False},
    {"id": "tablet", "width": 768, "height": 1024, "compact": True},
    {"id": "mobile", "width": 390, "height": 844, "compact": True},
    {"id": "mobile_narrow", "width": 360, "height": 800, "compact": True},
)
REQUIRED_SURFACE_STATE_VIEWPORTS: tuple[str, ...] = ("wide", "mobile")

_PLANNED_EVIDENCE = (
    "focus_indicator_audit",
    "contrast_audit",
    "surface_critic",
    "primary_job_simulation",
)


def _fingerprint(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_surface_quality_receipt(
    *,
    build_id: str,
    revision_id: str,
    source_checksum: str,
    bundle_sha256: str | None,
    validation_passed: bool,
    manifest: SurfaceManifest,
    runtime_proven: bool,
    runtime_diagnostics: list[dict[str, Any]],
    inspection_evidence: dict[str, Any],
    inspected_at: datetime | None = None,
) -> dict[str, Any]:
    """Create a content-bound receipt from trusted host inspection evidence."""
    checked_at = inspected_at or datetime.now(timezone.utc)
    interaction_required = bool(manifest.interaction_checks)
    viewport_matrix = dict(inspection_evidence.get("viewport_matrix") or {})
    viewports = list(viewport_matrix.get("viewports") or [])
    expected_viewports = [str(item["id"]) for item in REQUIRED_SURFACE_VIEWPORTS]
    observed_viewports = [
        str(item.get("id") or "") for item in viewports if isinstance(item, dict)
    ]
    viewport_passed = (
        viewport_matrix.get("passed") is True
        and viewport_matrix.get("required") == expected_viewports
        and observed_viewports == expected_viewports
        and all(
            isinstance(item, dict)
            and isinstance(item.get("capture"), dict)
            and bool(item["capture"].get("sha256"))
            for item in viewports
        )
    )
    accessibility_passed = viewport_passed and all(
        isinstance(item, dict)
        and isinstance(item.get("accessibility"), dict)
        and int(item["accessibility"].get("main_landmarks") or 0) == 1
        and int(item["accessibility"].get("unnamed_controls") or 0) == 0
        and int(item["accessibility"].get("images_missing_alt") or 0) == 0
        and int(item["accessibility"].get("keyboard_unreachable") or 0) == 0
        and not item["accessibility"].get("duplicate_ids")
        for item in viewports
    )
    state_matrix = dict(inspection_evidence.get("state_matrix") or {})
    state_observations = list(state_matrix.get("observations") or [])
    expected_states = list(REQUIRED_SURFACE_STATE_FIXTURES)
    expected_state_viewports = list(REQUIRED_SURFACE_STATE_VIEWPORTS)
    expected_combinations = [
        (state, viewport)
        for state in expected_states
        for viewport in expected_state_viewports
    ]
    observed_combinations = [
        (str(item.get("state") or ""), str(item.get("viewport_id") or ""))
        for item in state_observations
        if isinstance(item, dict)
    ]
    state_matrix_passed = (
        state_matrix.get("passed") is True
        and state_matrix.get("required_states") == expected_states
        and state_matrix.get("required_viewports") == expected_state_viewports
        and observed_combinations == expected_combinations
        and all(
            isinstance(item, dict)
            and isinstance(item.get("capture"), dict)
            and bool(item["capture"].get("sha256"))
            for item in state_observations
        )
    )
    checks: dict[str, dict[str, Any]] = {
        "deterministic_validation": {
            "status": "passed" if validation_passed else "failed",
        },
        "runtime_execution": {
            "status": "passed" if runtime_proven else "failed",
        },
        "declared_interactions": {
            "status": (
                "passed"
                if interaction_required and runtime_proven
                else "not_applicable" if not interaction_required else "failed"
            ),
            "declared": len(manifest.interaction_checks),
        },
        "viewport_matrix": {
            "status": "passed" if viewport_passed else "failed",
            "required": expected_viewports,
        },
        "accessibility_audit": {
            "status": "passed" if accessibility_passed else "failed",
            "scope": "deterministic_dom",
        },
        "state_matrix": {
            "status": "passed" if state_matrix_passed else "failed",
            "required_states": expected_states,
            "required_viewports": expected_state_viewports,
        },
    }
    passed = (
        validation_passed
        and runtime_proven
        and checks["declared_interactions"]["status"] in {"passed", "not_applicable"}
        and viewport_passed
        and accessibility_passed
        and state_matrix_passed
    )
    receipt: dict[str, Any] = {
        "policy_version": SURFACE_QUALITY_POLICY_VERSION,
        "passed": passed,
        "binding": {
            "build_id": build_id,
            "revision_id": revision_id,
            "source_checksum": source_checksum,
            "bundle_sha256": bundle_sha256,
        },
        "checks": checks,
        "evidence": inspection_evidence,
        "diagnostics": [dict(item) for item in runtime_diagnostics[:50]],
        "inspected_at": checked_at.isoformat(),
        # These remain explicit instead of being silently implied by a pass.
        # Later R9 policies promote each item into a verified check.
        "planned_evidence": list(_PLANNED_EVIDENCE),
    }
    receipt["fingerprint"] = _fingerprint(receipt)
    return receipt


def surface_quality_receipt_error(build: Any) -> str | None:
    """Return why an exact build lacks a valid passing host quality receipt."""
    receipt = dict(
        (build.resource_metrics or {}).get(SURFACE_QUALITY_RECEIPT_KEY) or {}
    )
    if not receipt:
        return "the exact build has no trusted quality receipt"
    fingerprint = str(receipt.pop("fingerprint", ""))
    if not fingerprint or fingerprint != _fingerprint(receipt):
        return "the quality receipt fingerprint is invalid"
    if receipt.get("policy_version") != SURFACE_QUALITY_POLICY_VERSION:
        return "the quality receipt uses an unsupported policy version"
    expected_binding = {
        "build_id": build.id,
        "revision_id": build.revision_id,
        "source_checksum": build.source_checksum,
        "bundle_sha256": build.bundle_sha256,
    }
    if receipt.get("binding") != expected_binding:
        return "the quality receipt belongs to different source or bundle content"
    if receipt.get("passed") is not True:
        return "the exact build did not pass the trusted quality gate"
    checks = dict(receipt.get("checks") or {})
    if dict(checks.get("deterministic_validation") or {}).get("status") != "passed":
        return "deterministic validation did not pass"
    if dict(checks.get("runtime_execution") or {}).get("status") != "passed":
        return "runtime inspection did not pass"
    if dict(checks.get("declared_interactions") or {}).get("status") not in {
        "passed",
        "not_applicable",
    }:
        return "declared interaction inspection did not pass"
    if dict(checks.get("viewport_matrix") or {}).get("status") != "passed":
        return "required viewport inspection did not pass"
    if dict(checks.get("accessibility_audit") or {}).get("status") != "passed":
        return "deterministic accessibility inspection did not pass"
    if dict(checks.get("state_matrix") or {}).get("status") != "passed":
        return "required Surface state inspection did not pass"
    return None


__all__ = [
    "SURFACE_QUALITY_POLICY_VERSION",
    "SURFACE_QUALITY_RECEIPT_KEY",
    "REQUIRED_SURFACE_VIEWPORTS",
    "REQUIRED_SURFACE_STATE_VIEWPORTS",
    "create_surface_quality_receipt",
    "surface_quality_receipt_error",
]
