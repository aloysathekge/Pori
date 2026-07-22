"""User-safe presentation for trusted Surface interaction lifecycles."""

from __future__ import annotations

import re
from typing import Any


def surface_request_label(
    *,
    declared_label: str | None = None,
    component_id: str | None = None,
) -> str:
    """Return a bounded human label without exposing command identifiers."""

    if declared_label and declared_label.strip():
        return declared_label.strip()[:160]
    words = re.sub(r"[._-]+", " ", (component_id or "").strip())
    words = re.sub(r"\s+", " ", words).strip()
    if words:
        return words[:160].capitalize()
    return "Continue from the Surface"


def interaction_presentation_label(
    result: dict[str, Any] | None,
    *,
    component_id: str | None,
) -> str:
    presentation = (result or {}).get("presentation")
    declared = (
        presentation.get("label")
        if isinstance(presentation, dict) and isinstance(presentation.get("label"), str)
        else None
    )
    return surface_request_label(
        declared_label=declared,
        component_id=component_id,
    )


__all__ = ["interaction_presentation_label", "surface_request_label"]
