"""Durable aggregation and user decisions for inferred Surface evolution."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from .models import (
    Event,
    EventTrailEntry,
    Run,
    SurfaceEvolutionProposal,
    SurfaceProject,
)
from .surface_evolution import (
    SurfaceEvolutionSignal,
    evaluate_surface_evolution,
    surface_evolution_signal_fingerprint,
)
from .surface_requests import (
    SURFACE_BUILDER_RUN_KIND,
    SurfaceRequestParams,
    queue_surface_builder_run,
)
from .tenancy import OrganizationContext

SURFACE_EVOLUTION_DISMISSAL_COOLDOWN_DAYS = 14


class SurfaceEvolutionProposalError(ValueError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _as_utc(value: datetime) -> datetime:
    """Normalize timestamps returned by databases that discard timezone metadata."""

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _datetime_payload(value: datetime | None) -> str | None:
    return _as_utc(value).isoformat() if value is not None else None


def surface_evolution_proposal_payload(
    proposal: SurfaceEvolutionProposal,
) -> dict:
    return {
        "id": proposal.id,
        "event_id": proposal.event_id,
        "project_id": proposal.project_id,
        "trigger": proposal.trigger,
        "goal": proposal.goal,
        "status": proposal.status,
        "occurrence_count": proposal.occurrence_count,
        "reason_codes": proposal.reason_codes,
        "evidence_refs": proposal.evidence_refs,
        "base_revision_id": proposal.base_revision_id,
        "base_build_id": proposal.base_build_id,
        "base_data_revision": proposal.base_data_revision,
        "builder_run_id": proposal.builder_run_id,
        "decided_by": proposal.decided_by,
        "decided_at": _datetime_payload(proposal.decided_at),
        "cooldown_until": _datetime_payload(proposal.cooldown_until),
        "created_at": _datetime_payload(proposal.created_at),
        "updated_at": _datetime_payload(proposal.updated_at),
    }


async def record_surface_evolution_signal(
    session: AsyncSession,
    *,
    context: OrganizationContext,
    event: Event,
    signal: SurfaceEvolutionSignal,
) -> SurfaceEvolutionProposal:
    """Aggregate one host-observed signal and expose it only when policy proposes."""

    project = (
        (
            await session.execute(
                select(SurfaceProject).where(
                    SurfaceProject.organization_id == context.organization_id,
                    SurfaceProject.user_id == context.user_id,
                    SurfaceProject.event_id == event.id,
                )
            )
        )
        .scalars()
        .first()
    )
    if (
        project is None
        or not project.published_revision_id
        or not project.published_build_id
    ):
        raise SurfaceEvolutionProposalError(
            409, "A published Surface is required before evolution can be proposed"
        )
    bound = signal.model_copy(
        update={
            "base_revision_id": project.published_revision_id,
            "base_build_id": project.published_build_id,
            "base_data_revision": project.data_revision,
            "event_archived": event.lifecycle == "archived",
        }
    )
    fingerprint = surface_evolution_signal_fingerprint(bound)
    existing = (
        (
            await session.execute(
                select(SurfaceEvolutionProposal).where(
                    SurfaceEvolutionProposal.event_id == event.id,
                    SurfaceEvolutionProposal.signal_fingerprint == fingerprint,
                )
            )
        )
        .scalars()
        .first()
    )
    now = datetime.now(timezone.utc)
    count = (existing.occurrence_count if existing else 0) + bound.occurrence_count
    decision = evaluate_surface_evolution(
        bound.model_copy(update={"occurrence_count": count})
    )
    evidence = list(existing.evidence_refs if existing else [])
    for reference in bound.evidence_refs:
        if reference not in evidence:
            evidence.append(reference)
    cooldown_active = bool(
        existing
        and existing.status == "dismissed"
        and existing.cooldown_until
        and _as_utc(existing.cooldown_until) > now
    )
    status = (
        "dismissed"
        if cooldown_active
        else "pending" if decision.outcome == "propose" else "observing"
    )
    proposal = existing or SurfaceEvolutionProposal(
        organization_id=context.organization_id,
        user_id=context.user_id,
        event_id=event.id,
        project_id=project.id,
        trigger=bound.trigger,
        goal=bound.goal,
        signal_fingerprint=fingerprint,
        decision_fingerprint=decision.fingerprint,
        base_revision_id=project.published_revision_id,
        base_build_id=project.published_build_id,
        base_data_revision=project.data_revision,
    )
    was_pending = proposal.status == "pending"
    proposal.occurrence_count = count
    proposal.decision_fingerprint = decision.fingerprint
    proposal.reason_codes = decision.reason_codes
    proposal.evidence_refs = evidence[:30]
    proposal.status = status
    if existing and existing.status == "dismissed" and not cooldown_active:
        proposal.decided_by = None
        proposal.decided_at = None
        proposal.cooldown_until = None
    proposal.updated_at = now
    session.add(proposal)
    if status == "pending" and not was_pending:
        session.add(
            EventTrailEntry(
                organization_id=context.organization_id,
                user_id=context.user_id,
                event_id=event.id,
                actor_id="surface-evolution-policy",
                kind="surface_evolution_proposed",
                summary="Suggested an improvement to the Event Surface",
                evidence_refs=[{"surface_evolution_proposal_id": proposal.id}],
                payload=surface_evolution_proposal_payload(proposal),
            )
        )
    await session.commit()
    await session.refresh(proposal)
    return proposal


async def decide_surface_evolution_proposal(
    session: AsyncSession,
    *,
    context: OrganizationContext,
    event: Event,
    proposal_id: str,
    decision: Literal["accept", "dismiss"],
) -> SurfaceEvolutionProposal:
    proposal = await session.get(SurfaceEvolutionProposal, proposal_id)
    if (
        proposal is None
        or proposal.organization_id != context.organization_id
        or proposal.user_id != context.user_id
        or proposal.event_id != event.id
    ):
        raise SurfaceEvolutionProposalError(404, "Surface evolution proposal not found")
    if proposal.status != "pending":
        raise SurfaceEvolutionProposalError(
            409, "Surface evolution proposal is not pending"
        )
    now = datetime.now(timezone.utc)
    if decision == "dismiss":
        proposal.status = "dismissed"
        proposal.decided_by = context.user_id
        proposal.decided_at = now
        proposal.cooldown_until = now + timedelta(
            days=SURFACE_EVOLUTION_DISMISSAL_COOLDOWN_DAYS
        )
    else:
        project = await session.get(SurfaceProject, proposal.project_id)
        if (
            project is None
            or project.published_revision_id != proposal.base_revision_id
            or project.published_build_id != proposal.base_build_id
        ):
            proposal.status = "superseded"
            proposal.updated_at = now
            session.add(proposal)
            await session.commit()
            raise SurfaceEvolutionProposalError(
                409, "The Surface changed after this proposal was created"
            )
        active = (
            (
                await session.execute(
                    select(Run).where(
                        Run.organization_id == context.organization_id,
                        Run.event_id == event.id,
                        Run.run_kind == SURFACE_BUILDER_RUN_KIND,
                        col(Run.status).in_(["pending", "running"]),
                    )
                )
            )
            .scalars()
            .first()
        )
        if active is not None:
            raise SurfaceEvolutionProposalError(
                409, "A Surface Builder is already active for this Event"
            )
        evolution = evaluate_surface_evolution(
            SurfaceEvolutionSignal(
                trigger="explicit_user_request",
                goal=proposal.goal,
                evidence_refs=[proposal.id, *proposal.evidence_refs],
                base_revision_id=proposal.base_revision_id,
                base_build_id=proposal.base_build_id,
                base_data_revision=project.data_revision,
            )
        )
        run = await queue_surface_builder_run(
            session,
            event=event,
            policy=context.policy,
            params=SurfaceRequestParams(
                goal=proposal.goal,
                experience=proposal.goal,
                jobs=[proposal.goal],
                source_refs=proposal.evidence_refs,
                interaction_notes=["Accepted Surface evolution proposal"],
            ),
            evolution=evolution,
            actor_id=context.user_id,
            parent_run_id=None,
            root_run_id=None,
            idempotency_key=f"surface-evolution:{proposal.id}",
            origin_evidence=[{"surface_evolution_proposal_id": proposal.id}],
        )
        proposal.status = "queued"
        proposal.builder_run_id = run.id
        proposal.decided_by = context.user_id
        proposal.decided_at = now
        proposal.cooldown_until = None
    proposal.updated_at = now
    session.add(proposal)
    session.add(
        EventTrailEntry(
            organization_id=context.organization_id,
            user_id=context.user_id,
            event_id=event.id,
            actor_id=context.user_id,
            kind="surface_evolution_decided",
            summary=(
                "Accepted a Surface improvement"
                if decision == "accept"
                else "Dismissed a Surface improvement"
            ),
            evidence_refs=[{"surface_evolution_proposal_id": proposal.id}],
            payload={
                "decision": decision,
                **surface_evolution_proposal_payload(proposal),
            },
        )
    )
    await session.commit()
    await session.refresh(proposal)
    return proposal


async def list_surface_evolution_proposals(
    session: AsyncSession,
    *,
    context: OrganizationContext,
    event_id: str,
) -> list[dict]:
    proposals = list(
        (
            await session.execute(
                select(SurfaceEvolutionProposal)
                .where(
                    SurfaceEvolutionProposal.organization_id == context.organization_id,
                    SurfaceEvolutionProposal.user_id == context.user_id,
                    SurfaceEvolutionProposal.event_id == event_id,
                    SurfaceEvolutionProposal.status == "pending",
                )
                .order_by(col(SurfaceEvolutionProposal.updated_at).desc())
            )
        )
        .scalars()
        .all()
    )
    return [surface_evolution_proposal_payload(item) for item in proposals]


__all__ = [
    "SURFACE_EVOLUTION_DISMISSAL_COOLDOWN_DAYS",
    "SurfaceEvolutionProposalError",
    "decide_surface_evolution_proposal",
    "list_surface_evolution_proposals",
    "record_surface_evolution_signal",
    "surface_evolution_proposal_payload",
]
