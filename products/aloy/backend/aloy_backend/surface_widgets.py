"""Host-reviewed widget registry for generated Event Surfaces.

The registry is deliberately small and domain-neutral.  A generated Surface
may name only widgets in this registry; privileged widgets are added only when
their host adapter and capability contract are ready.  The iframe receives the
declared ids, never provider credentials or arbitrary component code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

SURFACE_WIDGET_REGISTRY_VERSION = "1"


@dataclass(frozen=True)
class SurfaceWidgetDefinition:
    """Reviewed metadata used while validating a Surface manifest."""

    id: str
    label: str
    required_capabilities: frozenset[str] = frozenset()
    privileged: bool = False


# These are presentation primitives, not domain logic.  Map is intentionally
# absent until its host-owned adapter has explicit privacy, attribution,
# credential, and mobile contracts.
SURFACE_WIDGET_REGISTRY: dict[str, SurfaceWidgetDefinition] = {
    definition.id: definition
    for definition in (
        SurfaceWidgetDefinition("table", "Table"),
        SurfaceWidgetDefinition("form", "Form"),
        SurfaceWidgetDefinition("timeline", "Timeline"),
        SurfaceWidgetDefinition("chart", "Chart"),
        SurfaceWidgetDefinition("kanban", "Kanban board"),
        SurfaceWidgetDefinition(
            "approval", "Approval summary", frozenset({"proposals"})
        ),
        SurfaceWidgetDefinition("file_viewer", "File viewer", frozenset({"files"})),
    )
}


def validate_surface_widgets(
    values: Iterable[str], capabilities: Iterable[str] = ()
) -> list[str]:
    """Validate and normalize manifest widget ids.

    Unknown ids, duplicate ids, and missing capability grants fail closed.  The
    returned order is stable and preserves the Builder's declared order so the
    manifest remains useful as a deterministic receipt.
    """

    widgets = list(values)
    if len(widgets) != len(set(widgets)):
        raise ValueError("Surface widgets must be unique")
    granted = set(capabilities)
    for widget_id in widgets:
        definition = SURFACE_WIDGET_REGISTRY.get(widget_id)
        if definition is None:
            raise ValueError(f"Unsupported Surface widget: {widget_id}")
        missing = definition.required_capabilities.difference(granted)
        if missing:
            raise ValueError(
                f"Surface widget {widget_id} requires capabilities: "
                + ", ".join(sorted(missing))
            )
    return widgets


def surface_widget_definition(widget_id: str) -> SurfaceWidgetDefinition | None:
    """Return reviewed metadata for a widget id, if it exists."""

    return SURFACE_WIDGET_REGISTRY.get(widget_id)


__all__ = [
    "SURFACE_WIDGET_REGISTRY",
    "SURFACE_WIDGET_REGISTRY_VERSION",
    "SurfaceWidgetDefinition",
    "surface_widget_definition",
    "validate_surface_widgets",
]
