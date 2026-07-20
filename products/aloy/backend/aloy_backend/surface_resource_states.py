"""Host-owned resource state projection and trusted Surface state fixtures.

Generated Surface code receives data and state through the same public context.
The trusted inspector transforms that context to exercise failure and recovery
states; it does not expose an inspection-only flag that generated code can use
to special-case the publication gate.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from .surface_manifest import SurfaceManifest

SURFACE_RESOURCE_STATE_VERSION: Literal["1"] = "1"
SURFACE_STATE_POLICY_VERSION = "aloy-surface-states@2"
SurfaceResourceStatus = Literal[
    "loading",
    "ready",
    "empty",
    "stale",
    "error",
    "permission_denied",
    "pending",
    "indeterminate",
]
SurfaceResourceFixtureStatus = Literal[
    "loading",
    "empty",
    "stale",
    "error",
    "permission_denied",
    "pending",
    "indeterminate",
]
SurfaceInspectionFixture = (
    SurfaceResourceFixtureStatus
    | Literal[
        "long_content",
        "approval_required",
    ]
)

REQUIRED_SURFACE_RESOURCE_STATE_FIXTURES: tuple[SurfaceResourceFixtureStatus, ...] = (
    "loading",
    "empty",
    "stale",
    "error",
    "permission_denied",
    "pending",
    "indeterminate",
)
REQUIRED_SURFACE_SCENARIO_FIXTURES: tuple[SurfaceInspectionFixture, ...] = (
    "long_content",
    "approval_required",
)
REQUIRED_SURFACE_STATE_FIXTURES: tuple[SurfaceInspectionFixture, ...] = (
    *REQUIRED_SURFACE_RESOURCE_STATE_FIXTURES,
    *REQUIRED_SURFACE_SCENARIO_FIXTURES,
)

_LONG_CONTENT_ITEMS = 24
_FIXTURE_TIME = "2026-07-20T09:00:00Z"
_LONG_TEXT = (
    "This semester plan combines lecture preparation, reading notes, assignment "
    "milestones, revision questions, source references, and follow-up decisions so "
    "the student can understand the work, its evidence, and the next useful action. "
)

_STATE_MESSAGES: dict[SurfaceResourceStatus, str] = {
    "loading": "Aloy is loading this Event data.",
    "ready": "This Event data is current.",
    "empty": "No Event data exists here yet.",
    "stale": "This Event data may be out of date.",
    "error": "Aloy could not load this Event data.",
    "permission_denied": "This Event data is not available with the current access.",
    "pending": "Aloy is updating this Event data.",
    "indeterminate": "The latest outcome is not yet known.",
}


def _resource_value(data: dict[str, Any], resource: str) -> Any:
    if resource.startswith("data:"):
        namespace = resource.removeprefix("data:")
        return dict(data.get("surface") or {}).get(namespace)
    if resource.startswith("records:"):
        namespace = resource.removeprefix("records:")
        return dict(data.get("records") or {}).get(namespace)
    return data.get(resource)


def _has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, dict, str)):
        return bool(value)
    return True


def _resource_capabilities(capabilities: list[Any]) -> list[str]:
    """Return data-bearing capabilities; action-only grants are not resources."""
    return [str(item) for item in capabilities if str(item) != "ask_aloy"]


def surface_resource_states(
    manifest: SurfaceManifest,
    data: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Describe the settled state of every capability-scoped resource."""
    return {
        resource: {
            "status": (
                "ready" if _has_content(_resource_value(data, resource)) else "empty"
            ),
            "message": _STATE_MESSAGES[
                "ready" if _has_content(_resource_value(data, resource)) else "empty"
            ],
            "retryable": False,
        }
        for resource in _resource_capabilities(manifest.capabilities)
    }


def _clear_resource(data: dict[str, Any], resource: str) -> None:
    if resource.startswith("data:"):
        namespace = resource.removeprefix("data:")
        surface = dict(data.get("surface") or {})
        surface[namespace] = []
        data["surface"] = surface
        return
    if resource.startswith("records:"):
        namespace = resource.removeprefix("records:")
        records = dict(data.get("records") or {})
        records[namespace] = []
        data["records"] = records
        return
    if resource == "event":
        data.pop("event", None)
        return
    data[resource] = []


def _project_resource_states(
    projected: dict[str, Any],
    data: dict[str, Any],
    resources: list[str],
) -> None:
    projected["resource_state_version"] = SURFACE_RESOURCE_STATE_VERSION
    projected["resource_states"] = {
        resource: {
            "status": (
                "ready" if _has_content(_resource_value(data, resource)) else "empty"
            ),
            "message": _STATE_MESSAGES[
                "ready" if _has_content(_resource_value(data, resource)) else "empty"
            ],
            "retryable": False,
        }
        for resource in resources
    }


def _long_content_fixture(data: dict[str, Any], resources: list[str]) -> None:
    """Populate only public capability-scoped shapes with dense deterministic data."""
    if "event" in resources:
        event = dict(data.get("event") or {})
        event.update(
            {
                "id": str(event.get("id") or "evt_university_2026"),
                "title": f"University planning and academic progress — {_LONG_TEXT}",
                "summary": _LONG_TEXT * 4,
                "updated_at": _FIXTURE_TIME,
            }
        )
        data["event"] = event
    if "tasks" in resources:
        data["tasks"] = [
            {
                "id": f"task_semester_{index + 1}",
                "event_id": "evt_university_2026",
                "title": f"Prepare detailed milestone {index + 1}: {_LONG_TEXT}",
                "status": "open",
                "instructions": _LONG_TEXT * 3,
                "definition_of_done": _LONG_TEXT * 2,
                "priority": "normal",
                "due_at": _FIXTURE_TIME,
                "order": index,
                "created_at": _FIXTURE_TIME,
                "updated_at": _FIXTURE_TIME,
            }
            for index in range(_LONG_CONTENT_ITEMS)
        ]
    if "files" in resources:
        data["files"] = [
            {
                "id": f"file_course_material_{index + 1}",
                "name": f"semester-research-and-supporting-evidence-{index + 1}.pdf",
                "kind": "document",
                "content_type": "application/pdf",
                "size_bytes": 2_000_000 + index,
                "created_at": _FIXTURE_TIME,
            }
            for index in range(_LONG_CONTENT_ITEMS)
        ]
    if "proposals" in resources:
        data["proposals"] = [
            {
                "id": f"proposal_calendar_archive_{index + 1}",
                "event_id": "evt_university_2026",
                "tool": "calendar_create_event",
                "args": {"title": f"Detailed calendar item {index + 1}"},
                "reason": _LONG_TEXT * 2,
                "impact": _LONG_TEXT,
                "risk": "low",
                "routing": "ask",
                "status": "rejected",
                "expires_at": None,
                "decided_at": _FIXTURE_TIME,
                "provider_operation_id": None,
                "receipt": None,
                "error": None,
                "created_at": _FIXTURE_TIME,
                "updated_at": _FIXTURE_TIME,
            }
            for index in range(_LONG_CONTENT_ITEMS)
        ]
    if "receipts" in resources:
        data["receipts"] = [
            {
                "proposal_id": f"proposal_calendar_committed_{index + 1}",
                "tool": "calendar_create_event",
                "receipt": {"summary": _LONG_TEXT, "provider_id": f"provider-{index}"},
                "status": "committed",
                "updated_at": _FIXTURE_TIME,
            }
            for index in range(_LONG_CONTENT_ITEMS)
        ]
    if "trail" in resources:
        data["trail"] = [
            {
                "id": f"trail_semester_update_{index + 1}",
                "kind": "event_updated",
                "summary": f"Detailed Event history item {index + 1}: {_LONG_TEXT}",
                "actor_id": "aloy",
                "run_id": None,
                "proposal_id": None,
                "task_id": None,
                "evidence_refs": [],
                "payload": {},
                "created_at": _FIXTURE_TIME,
            }
            for index in range(_LONG_CONTENT_ITEMS)
        ]
    for resource in resources:
        if resource.startswith("data:"):
            namespace = resource.removeprefix("data:")
            surface = dict(data.get("surface") or {})
            surface[namespace] = [
                {
                    "id": f"surface_{namespace}_{index + 1}",
                    "namespace": namespace,
                    "key": f"semester-item-{index + 1}",
                    "data": {
                        "title": f"Detailed {namespace} record {index + 1}",
                        "summary": _LONG_TEXT * 2,
                    },
                    "revision": 1,
                    "posture": "user_reported",
                    "provenance": {"source": "event_context"},
                    "evidence_refs": [],
                }
                for index in range(_LONG_CONTENT_ITEMS)
            ]
            data["surface"] = surface
        if resource.startswith("records:"):
            namespace = resource.removeprefix("records:")
            records = dict(data.get("records") or {})
            records[namespace] = [
                {
                    "id": f"record_{namespace}_{index + 1}",
                    "namespace": namespace,
                    "key": f"evidence-item-{index + 1}",
                    "title": f"Evidence-backed {namespace} record {index + 1}",
                    "summary": _LONG_TEXT * 2,
                    "data": {"notes": _LONG_TEXT * 2},
                    "posture": "observed",
                    "confidence": 1.0,
                    "revision": 1,
                    "evidence_refs": [],
                    "created_at": _FIXTURE_TIME,
                    "updated_at": _FIXTURE_TIME,
                }
                for index in range(_LONG_CONTENT_ITEMS)
            ]
            data["records"] = records


def _approval_fixture(
    projected: dict[str, Any],
    data: dict[str, Any],
    manifest: SurfaceManifest | None,
) -> None:
    capabilities = set(
        _resource_capabilities(list(projected.get("capabilities") or []))
    )
    external_action = bool(
        manifest
        and any(
            declaration.interaction_class == "external_action"
            for declaration in manifest.intents.values()
        )
    )
    if "proposals" not in capabilities and not external_action:
        return
    proposal_id = "proposal_calendar_exam_review"
    if "proposals" in capabilities:
        data["proposals"] = [
            {
                "id": proposal_id,
                "event_id": str(projected.get("event_id") or "evt_university_2026"),
                "tool": "calendar_create_event",
                "args": {"title": "Protected calendar change"},
                "reason": "Aloy prepared this change from the Event plan.",
                "impact": "Creates an external calendar event after approval.",
                "risk": "high",
                "routing": "ask",
                "status": "pending",
                "expires_at": None,
                "decided_at": None,
                "provider_operation_id": None,
                "receipt": None,
                "error": None,
                "created_at": _FIXTURE_TIME,
                "updated_at": _FIXTURE_TIME,
            }
        ]
    interactions = list(data.get("interactions") or [])
    interactions.insert(
        0,
        {
            "id": "interaction_calendar_exam_review",
            "event_id": str(projected.get("event_id") or "evt_university_2026"),
            "build_id": str(projected.get("build_id") or "build_candidate"),
            "code_revision_id": str(
                projected.get("code_revision_id") or "revision_candidate"
            ),
            "name": "calendar.event_create",
            "interaction_class": "external_action",
            "component_id": "add-exam-to-calendar",
            "status": "waiting_approval",
            "handling_run_id": "run_calendar_exam_review",
            "proposal_id": proposal_id,
            "request_message_id": None,
            "outcome_message_id": None,
            "result": {},
            "error": None,
            "created_at": _FIXTURE_TIME,
            "updated_at": _FIXTURE_TIME,
        },
    )
    data["interactions"] = interactions


def surface_fixture_applicable(
    manifest: SurfaceManifest | None,
    fixture: SurfaceInspectionFixture,
    capabilities: list[Any],
) -> bool:
    resources = _resource_capabilities(capabilities)
    if fixture in REQUIRED_SURFACE_RESOURCE_STATE_FIXTURES:
        return bool(resources)
    if fixture == "long_content":
        return bool(resources)
    return "proposals" in resources or bool(
        manifest
        and any(
            declaration.interaction_class == "external_action"
            for declaration in manifest.intents.values()
        )
    )


def surface_state_fixture_context(
    context: dict[str, Any],
    status: SurfaceInspectionFixture,
    *,
    manifest: SurfaceManifest | None = None,
) -> dict[str, Any]:
    """Project one real public state for trusted browser inspection."""
    if status not in REQUIRED_SURFACE_STATE_FIXTURES:
        raise ValueError(f"Unsupported Surface state fixture: {status}")
    projected = deepcopy(context)
    data = dict(projected.get("data") or {})
    resource_states = dict(projected.get("resource_states") or {})
    resources = sorted(
        set(_resource_capabilities(list(resource_states)))
        | set(_resource_capabilities(list(projected.get("capabilities") or [])))
    )

    if status == "long_content":
        _long_content_fixture(data, resources)
        _project_resource_states(projected, data, resources)
        projected["data"] = data
        return projected
    if status == "approval_required":
        _approval_fixture(projected, data, manifest)
        _project_resource_states(projected, data, resources)
        projected["data"] = data
        return projected

    if status in {"loading", "empty", "error", "permission_denied"}:
        for resource in resources:
            _clear_resource(data, resource)

    retryable = status in {"loading", "stale", "error", "pending", "indeterminate"}
    projected["resource_state_version"] = SURFACE_RESOURCE_STATE_VERSION
    projected["resource_states"] = {
        resource: {
            "status": status,
            "message": _STATE_MESSAGES[status],
            "retryable": retryable,
        }
        for resource in resources
    }
    projected["data"] = data
    return projected


__all__ = [
    "REQUIRED_SURFACE_RESOURCE_STATE_FIXTURES",
    "REQUIRED_SURFACE_SCENARIO_FIXTURES",
    "REQUIRED_SURFACE_STATE_FIXTURES",
    "SURFACE_RESOURCE_STATE_VERSION",
    "SURFACE_STATE_POLICY_VERSION",
    "SurfaceInspectionFixture",
    "SurfaceResourceFixtureStatus",
    "SurfaceResourceStatus",
    "surface_fixture_applicable",
    "surface_resource_states",
    "surface_state_fixture_context",
]
