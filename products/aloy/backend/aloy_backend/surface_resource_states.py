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

REQUIRED_SURFACE_STATE_FIXTURES: tuple[SurfaceResourceStatus, ...] = (
    "loading",
    "empty",
    "stale",
    "error",
    "permission_denied",
    "pending",
    "indeterminate",
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
        for resource in manifest.capabilities
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


def surface_state_fixture_context(
    context: dict[str, Any],
    status: SurfaceResourceStatus,
) -> dict[str, Any]:
    """Project one real public state for trusted browser inspection."""
    if status not in REQUIRED_SURFACE_STATE_FIXTURES:
        raise ValueError(f"Unsupported Surface state fixture: {status}")
    projected = deepcopy(context)
    data = dict(projected.get("data") or {})
    resource_states = dict(projected.get("resource_states") or {})
    resources = list(resource_states)
    if not resources:
        resources = [str(item) for item in projected.get("capabilities") or []]

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
    "REQUIRED_SURFACE_STATE_FIXTURES",
    "SURFACE_RESOURCE_STATE_VERSION",
    "SurfaceResourceStatus",
    "surface_resource_states",
    "surface_state_fixture_context",
]
