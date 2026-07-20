"""Host-enforced context gate for Runs started by generated Surfaces."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

SURFACE_INTERACTION_CONTEXT_GATE_VERSION = "1"


@dataclass(frozen=True)
class SurfaceInteractionContextGateResult:
    """Evidence that a Surface Run read its exact originating interaction."""

    required_interaction_id: str
    observed_interaction_ids: tuple[str, ...]
    accepted: bool
    errors: tuple[str, ...]

    def receipt(self) -> dict[str, Any]:
        return {
            "kind": "surface_interaction_context_gate",
            "version": SURFACE_INTERACTION_CONTEXT_GATE_VERSION,
            "accepted": self.accepted,
            "required_interaction_id": self.required_interaction_id,
            "observed_interaction_ids": list(self.observed_interaction_ids),
            "errors": list(self.errors),
        }


def evaluate_surface_interaction_context(
    *,
    required_interaction_id: str,
    observed_interaction_ids: Collection[str],
) -> SurfaceInteractionContextGateResult:
    """Fail closed unless the Run resolved the exact trusted interaction id."""
    observed = tuple(sorted(set(observed_interaction_ids)))
    accepted = required_interaction_id in observed
    errors = (
        ()
        if accepted
        else ("The Run did not read its exact originating Surface interaction.",)
    )
    return SurfaceInteractionContextGateResult(
        required_interaction_id=required_interaction_id,
        observed_interaction_ids=observed,
        accepted=accepted,
        errors=errors,
    )


__all__ = [
    "SURFACE_INTERACTION_CONTEXT_GATE_VERSION",
    "SurfaceInteractionContextGateResult",
    "evaluate_surface_interaction_context",
]
