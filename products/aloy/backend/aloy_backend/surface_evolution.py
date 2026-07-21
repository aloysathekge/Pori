"""Deterministic host policy for deciding when a Surface should evolve."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SURFACE_EVOLUTION_POLICY_VERSION: Literal["aloy-surface-evolution@1"] = (
    "aloy-surface-evolution@1"
)

SurfaceEvolutionTrigger = Literal[
    "explicit_user_request",
    "surface_source_change",
    "event_phase_changed",
    "capability_gap",
    "negative_feedback",
    "primary_job_failure",
    "quality_failure",
]
SurfaceEvolutionOutcome = Literal["queue", "propose", "ignore"]


class SurfaceEvolutionSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trigger: SurfaceEvolutionTrigger
    goal: str = Field(min_length=3, max_length=1000)
    evidence_refs: list[str] = Field(default_factory=list, max_length=30)
    occurrence_count: int = Field(default=1, ge=1, le=10_000)
    base_revision_id: str | None = Field(default=None, max_length=200)
    base_build_id: str | None = Field(default=None, max_length=200)
    base_data_revision: int = Field(default=0, ge=0)
    active_builder: bool = False
    event_archived: bool = False


class SurfaceEvolutionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_version: Literal["aloy-surface-evolution@1"] = (
        SURFACE_EVOLUTION_POLICY_VERSION
    )
    outcome: SurfaceEvolutionOutcome
    reason_codes: list[str]
    trigger: SurfaceEvolutionTrigger
    goal: str
    base_revision_id: str | None
    base_build_id: str | None
    base_data_revision: int
    fingerprint: str

    def receipt(self) -> dict:
        return {"kind": "surface_evolution_decision", **self.model_dump(mode="json")}


def evaluate_surface_evolution(
    signal: SurfaceEvolutionSignal,
) -> SurfaceEvolutionDecision:
    """Return a reproducible queue/propose/ignore decision.

    Explicit user intent and a published Surface's declared source-change
    control may queue work. Inferred product signals only propose evolution;
    they never silently redesign a familiar workspace.
    """

    reasons: list[str] = []
    if signal.event_archived:
        outcome: SurfaceEvolutionOutcome = "ignore"
        reasons.append("event_archived")
    elif signal.active_builder:
        outcome = "ignore"
        reasons.append("builder_already_active")
    elif signal.trigger in {"explicit_user_request", "surface_source_change"}:
        outcome = "queue"
        reasons.append("explicit_source_change")
    elif signal.trigger == "quality_failure":
        outcome = "propose"
        reasons.append("trusted_quality_failure")
    elif signal.trigger == "primary_job_failure" and signal.occurrence_count >= 2:
        outcome = "propose"
        reasons.append("repeated_primary_job_failure")
    elif signal.trigger in {
        "event_phase_changed",
        "capability_gap",
        "negative_feedback",
    }:
        outcome = "propose"
        reasons.append(signal.trigger)
    else:
        outcome = "ignore"
        reasons.append("insufficient_evidence")

    body = {
        "policy_version": SURFACE_EVOLUTION_POLICY_VERSION,
        "outcome": outcome,
        "reason_codes": reasons,
        "trigger": signal.trigger,
        "goal": " ".join(signal.goal.split()),
        "base_revision_id": signal.base_revision_id,
        "base_build_id": signal.base_build_id,
        "base_data_revision": signal.base_data_revision,
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return SurfaceEvolutionDecision(
        outcome=outcome,
        reason_codes=reasons,
        trigger=signal.trigger,
        goal=str(body["goal"]),
        base_revision_id=signal.base_revision_id,
        base_build_id=signal.base_build_id,
        base_data_revision=signal.base_data_revision,
        fingerprint=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    )


__all__ = [
    "SURFACE_EVOLUTION_POLICY_VERSION",
    "SurfaceEvolutionDecision",
    "SurfaceEvolutionSignal",
    "evaluate_surface_evolution",
]
