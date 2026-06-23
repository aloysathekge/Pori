"""Governed self-evolution contracts.

Release E starts with proposals, eval evidence, approval, activation, and
rollback as explicit state transitions. Nothing here mutates skills, prompts,
or code by itself.
"""

from __future__ import annotations

import json
import uuid
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .runtime import stable_fingerprint, utc_now


class EvolutionArtifactKind(str, Enum):
    SKILL = "skill"
    PROMPT = "prompt"
    POLICY = "policy"
    EVAL = "eval"
    CODE = "code"
    CONFIG = "config"


class EvolutionProposalStatus(str, Enum):
    PROPOSED = "proposed"
    EVALUATED = "evaluated"
    APPROVED = "approved"
    REJECTED = "rejected"
    ACTIVATED = "activated"
    ROLLED_BACK = "rolled_back"


class EvolutionEvalCase(BaseModel):
    """Required evaluation evidence a proposal must satisfy."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=120)
    input: str = Field(min_length=1)
    expected: Optional[str] = None
    criteria: str = Field(min_length=1, max_length=1000)


class EvolutionEvalResult(BaseModel):
    """Observed result for one proposal evaluation."""

    model_config = ConfigDict(frozen=True)

    case_name: str = Field(min_length=1, max_length=120)
    passed: bool
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=2000)
    evaluator: str = Field(default="manual", min_length=1, max_length=120)
    evaluated_at: str = Field(default_factory=lambda: utc_now().isoformat())


class EvolutionProposal(BaseModel):
    """A proposed self-evolution change that is inert until approved."""

    model_config = ConfigDict(frozen=True)

    proposal_id: str = Field(
        default_factory=lambda: f"evo_{uuid.uuid4().hex[:16]}",
        min_length=1,
    )
    artifact_kind: EvolutionArtifactKind
    target: str = Field(min_length=1, max_length=240)
    title: str = Field(min_length=1, max_length=240)
    summary: str = Field(min_length=1, max_length=1000)
    rationale: str = Field(min_length=1, max_length=4000)
    proposed_version: str = Field(min_length=1, max_length=80)
    current_version: Optional[str] = Field(default=None, max_length=80)
    proposed_content: str = Field(min_length=1)
    eval_cases: Tuple[EvolutionEvalCase, ...]
    eval_results: Tuple[EvolutionEvalResult, ...] = ()
    status: EvolutionProposalStatus = EvolutionProposalStatus.PROPOSED
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())
    approved_by: Optional[str] = None
    activated_at: Optional[str] = None
    supersedes_proposal_id: Optional[str] = None

    @model_validator(mode="after")
    def require_eval_cases(self) -> "EvolutionProposal":
        if not self.eval_cases:
            raise ValueError("Evolution proposals require at least one eval case")
        return self

    @property
    def content_fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "artifact_kind": self.artifact_kind.value,
                "target": self.target,
                "version": self.proposed_version,
                "content": self.proposed_content,
            }
        )

    @property
    def evaluations_passed(self) -> bool:
        expected = {case.name for case in self.eval_cases}
        observed = {result.case_name for result in self.eval_results if result.passed}
        return expected.issubset(observed)


class EvolutionActivation(BaseModel):
    """Version activation record for one target."""

    model_config = ConfigDict(frozen=True)

    target: str
    proposal_id: str
    version: str
    activated_by: str
    activated_at: str = Field(default_factory=lambda: utc_now().isoformat())
    rolled_back_at: Optional[str] = None


class EvolutionRepository:
    """In-memory governed proposal repository.

    Storage adapters can implement the same transition rules later. This first
    repository is intentionally small and deterministic for local CLI/runtime use.
    """

    def __init__(self):
        self._proposals: Dict[str, EvolutionProposal] = {}
        self._active_by_target: Dict[str, EvolutionActivation] = {}
        self._history_by_target: Dict[str, List[EvolutionActivation]] = {}

    def submit(self, proposal: EvolutionProposal) -> EvolutionProposal:
        if proposal.proposal_id in self._proposals:
            raise ValueError(f"Proposal '{proposal.proposal_id}' already exists")
        self._proposals[proposal.proposal_id] = proposal
        return proposal

    def get(self, proposal_id: str) -> EvolutionProposal:
        try:
            return self._proposals[proposal_id]
        except KeyError as exc:
            raise ValueError(f"Unknown proposal '{proposal_id}'") from exc

    def list(
        self,
        *,
        status: Optional[EvolutionProposalStatus] = None,
        target: Optional[str] = None,
    ) -> Tuple[EvolutionProposal, ...]:
        proposals: Iterable[EvolutionProposal] = self._proposals.values()
        if status is not None:
            proposals = [
                proposal for proposal in proposals if proposal.status == status
            ]
        if target is not None:
            proposals = [
                proposal for proposal in proposals if proposal.target == target
            ]
        return tuple(sorted(proposals, key=lambda proposal: proposal.created_at))

    def record_evaluations(
        self,
        proposal_id: str,
        results: Iterable[EvolutionEvalResult],
    ) -> EvolutionProposal:
        proposal = self.get(proposal_id)
        self._ensure_status(
            proposal,
            EvolutionProposalStatus.PROPOSED,
            EvolutionProposalStatus.EVALUATED,
        )
        updated = proposal.model_copy(
            update={
                "eval_results": tuple(results),
                "status": EvolutionProposalStatus.EVALUATED,
            }
        )
        self._proposals[proposal_id] = updated
        return updated

    def approve(self, proposal_id: str, *, reviewer: str) -> EvolutionProposal:
        proposal = self.get(proposal_id)
        self._ensure_status(proposal, EvolutionProposalStatus.EVALUATED)
        if not proposal.evaluations_passed:
            raise ValueError("Proposal cannot be approved until all eval cases pass")
        updated = proposal.model_copy(
            update={
                "status": EvolutionProposalStatus.APPROVED,
                "approved_by": reviewer,
            }
        )
        self._proposals[proposal_id] = updated
        return updated

    def reject(self, proposal_id: str, *, reviewer: str) -> EvolutionProposal:
        proposal = self.get(proposal_id)
        self._ensure_status(
            proposal,
            EvolutionProposalStatus.PROPOSED,
            EvolutionProposalStatus.EVALUATED,
        )
        updated = proposal.model_copy(
            update={
                "status": EvolutionProposalStatus.REJECTED,
                "approved_by": reviewer,
            }
        )
        self._proposals[proposal_id] = updated
        return updated

    def activate(self, proposal_id: str, *, activated_by: str) -> EvolutionActivation:
        proposal = self.get(proposal_id)
        self._ensure_status(proposal, EvolutionProposalStatus.APPROVED)
        activation = EvolutionActivation(
            target=proposal.target,
            proposal_id=proposal.proposal_id,
            version=proposal.proposed_version,
            activated_by=activated_by,
        )
        self._active_by_target[proposal.target] = activation
        self._history_by_target.setdefault(proposal.target, []).append(activation)
        self._proposals[proposal_id] = proposal.model_copy(
            update={
                "status": EvolutionProposalStatus.ACTIVATED,
                "activated_at": activation.activated_at,
            }
        )
        return activation

    def active(self, target: str) -> Optional[EvolutionActivation]:
        return self._active_by_target.get(target)

    def rollback(
        self, target: str, *, rolled_back_by: str
    ) -> Optional[EvolutionActivation]:
        current = self._active_by_target.get(target)
        if current is None:
            return None
        rolled_back = current.model_copy(
            update={"rolled_back_at": utc_now().isoformat()}
        )
        self._history_by_target[target][-1] = rolled_back
        self._proposals[current.proposal_id] = self.get(current.proposal_id).model_copy(
            update={"status": EvolutionProposalStatus.ROLLED_BACK}
        )
        prior = [
            item
            for item in self._history_by_target.get(target, [])[:-1]
            if item.rolled_back_at is None
        ]
        if prior:
            restored = prior[-1]
            self._active_by_target[target] = restored
            return restored
        self._active_by_target.pop(target, None)
        return None

    def snapshot(self) -> Dict[str, object]:
        return {
            "proposals": [
                proposal.model_dump(mode="json")
                for proposal in sorted(
                    self._proposals.values(), key=lambda item: item.created_at
                )
            ],
            "active_by_target": {
                target: activation.model_dump(mode="json")
                for target, activation in sorted(self._active_by_target.items())
            },
            "history_by_target": {
                target: [
                    activation.model_dump(mode="json") for activation in activations
                ]
                for target, activations in sorted(self._history_by_target.items())
            },
        }

    @classmethod
    def from_snapshot(cls, payload: Dict[str, object]) -> "EvolutionRepository":
        repo = cls()
        proposals = payload.get("proposals", [])
        if isinstance(proposals, list):
            for raw in proposals:
                if isinstance(raw, dict):
                    proposal = EvolutionProposal(**raw)
                    repo._proposals[proposal.proposal_id] = proposal
        active_by_target = payload.get("active_by_target", {})
        if isinstance(active_by_target, dict):
            for target, raw in active_by_target.items():
                if isinstance(raw, dict):
                    repo._active_by_target[str(target)] = EvolutionActivation(**raw)
        history_by_target = payload.get("history_by_target", {})
        if isinstance(history_by_target, dict):
            for target, activations in history_by_target.items():
                if isinstance(activations, list):
                    repo._history_by_target[str(target)] = [
                        EvolutionActivation(**raw)
                        for raw in activations
                        if isinstance(raw, dict)
                    ]
        return repo

    @staticmethod
    def _ensure_status(
        proposal: EvolutionProposal,
        *allowed: EvolutionProposalStatus,
    ) -> None:
        if proposal.status not in allowed:
            allowed_values = ", ".join(status.value for status in allowed)
            raise ValueError(
                f"Proposal '{proposal.proposal_id}' is {proposal.status.value}; "
                f"expected one of: {allowed_values}"
            )


class FileEvolutionRepository(EvolutionRepository):
    """JSON-backed governed proposal repository."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                loaded = EvolutionRepository.from_snapshot(payload)
                self._proposals = dict(loaded._proposals)
                self._active_by_target = dict(loaded._active_by_target)
                self._history_by_target = {
                    key: list(value) for key, value in loaded._history_by_target.items()
                }
            except Exception:
                super().__init__()
        else:
            super().__init__()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.snapshot(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def submit(self, proposal: EvolutionProposal) -> EvolutionProposal:
        result = super().submit(proposal)
        self.save()
        return result

    def record_evaluations(
        self,
        proposal_id: str,
        results: Iterable[EvolutionEvalResult],
    ) -> EvolutionProposal:
        result = super().record_evaluations(proposal_id, results)
        self.save()
        return result

    def approve(self, proposal_id: str, *, reviewer: str) -> EvolutionProposal:
        result = super().approve(proposal_id, reviewer=reviewer)
        self.save()
        return result

    def reject(self, proposal_id: str, *, reviewer: str) -> EvolutionProposal:
        result = super().reject(proposal_id, reviewer=reviewer)
        self.save()
        return result

    def activate(self, proposal_id: str, *, activated_by: str) -> EvolutionActivation:
        result = super().activate(proposal_id, activated_by=activated_by)
        self.save()
        return result

    def rollback(
        self, target: str, *, rolled_back_by: str
    ) -> Optional[EvolutionActivation]:
        result = super().rollback(target, rolled_back_by=rolled_back_by)
        self.save()
        return result


__all__ = [
    "EvolutionActivation",
    "EvolutionArtifactKind",
    "EvolutionEvalCase",
    "EvolutionEvalResult",
    "FileEvolutionRepository",
    "EvolutionProposal",
    "EvolutionProposalStatus",
    "EvolutionRepository",
]
