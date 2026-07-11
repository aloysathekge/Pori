"""Evolution endpoints: org-scoped CRUD for ``EvolutionProposal`` rows and
their lifecycle (eval recording, approve/reject, activation via
``EvolutionActivation``).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from ..database import get_session
from ..models import EvolutionActivation, EvolutionProposal
from ..schemas import (
    EvolutionActivationResponse,
    EvolutionEvalRecord,
    EvolutionProposalCreate,
    EvolutionProposalResponse,
)
from ..tenancy import OrganizationContext, Permission, require_permission

router = APIRouter(prefix="/evolution", tags=["evolution"])


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def _get_proposal(
    session: AsyncSession,
    proposal_id: str,
    organization_id: str,
) -> EvolutionProposal:
    proposal = await session.get(EvolutionProposal, proposal_id)
    if not proposal or proposal.organization_id != organization_id:
        raise HTTPException(status_code=404, detail="Evolution proposal not found")
    return proposal


def _evals_passed(proposal: EvolutionProposal) -> bool:
    expected = {
        str(item.get("name"))
        for item in proposal.eval_cases
        if isinstance(item, dict) and item.get("name")
    }
    observed = {
        str(item.get("case_name"))
        for item in proposal.eval_results
        if isinstance(item, dict) and item.get("passed") is True
    }
    return bool(expected) and expected.issubset(observed)


def _require_status(proposal: EvolutionProposal, *allowed: str) -> None:
    if proposal.status not in allowed:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Proposal is {proposal.status}; expected one of: " + ", ".join(allowed)
            ),
        )


@router.post("", response_model=EvolutionProposalResponse, status_code=201)
async def create_evolution_proposal(
    body: EvolutionProposalCreate,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> EvolutionProposal:
    proposal = EvolutionProposal(
        organization_id=context.organization_id,
        created_by=context.user_id,
        eval_cases=[item.model_dump(mode="json") for item in body.eval_cases],
        **body.model_dump(
            exclude={"eval_cases"},
            mode="json",
        ),
    )
    session.add(proposal)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=409, detail="Evolution proposal exists"
        ) from exc
    await session.refresh(proposal)
    return proposal


@router.get("", response_model=list[EvolutionProposalResponse])
async def list_evolution_proposals(
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
) -> Sequence[EvolutionProposal]:
    result = await session.execute(
        select(EvolutionProposal)
        .where(EvolutionProposal.organization_id == context.organization_id)
        .order_by(col(EvolutionProposal.created_at).desc())
    )
    return result.scalars().all()


@router.get("/{proposal_id}", response_model=EvolutionProposalResponse)
async def get_evolution_proposal(
    proposal_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
) -> EvolutionProposal:
    return await _get_proposal(session, proposal_id, context.organization_id)


@router.post("/{proposal_id}/evals", response_model=EvolutionProposalResponse)
async def record_evolution_evals(
    proposal_id: str,
    body: EvolutionEvalRecord,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> EvolutionProposal:
    proposal = await _get_proposal(session, proposal_id, context.organization_id)
    _require_status(proposal, "proposed", "evaluated")
    proposal.eval_results = [item.model_dump(mode="json") for item in body.results]
    proposal.status = "evaluated"
    proposal.updated_at = _utcnow()
    session.add(proposal)
    await session.commit()
    await session.refresh(proposal)
    return proposal


@router.post("/{proposal_id}/approve", response_model=EvolutionProposalResponse)
async def approve_evolution_proposal(
    proposal_id: str,
    context: OrganizationContext = Depends(
        require_permission(Permission.POLICY_MANAGE)
    ),
    session: AsyncSession = Depends(get_session),
) -> EvolutionProposal:
    proposal = await _get_proposal(session, proposal_id, context.organization_id)
    _require_status(proposal, "evaluated")
    if not _evals_passed(proposal):
        raise HTTPException(
            status_code=409,
            detail="Proposal cannot be approved until all eval cases pass",
        )
    proposal.status = "approved"
    proposal.approved_by = context.user_id
    proposal.updated_at = _utcnow()
    session.add(proposal)
    await session.commit()
    await session.refresh(proposal)
    return proposal


@router.post("/{proposal_id}/reject", response_model=EvolutionProposalResponse)
async def reject_evolution_proposal(
    proposal_id: str,
    context: OrganizationContext = Depends(
        require_permission(Permission.POLICY_MANAGE)
    ),
    session: AsyncSession = Depends(get_session),
) -> EvolutionProposal:
    proposal = await _get_proposal(session, proposal_id, context.organization_id)
    _require_status(proposal, "proposed", "evaluated")
    proposal.status = "rejected"
    proposal.approved_by = context.user_id
    proposal.updated_at = _utcnow()
    session.add(proposal)
    await session.commit()
    await session.refresh(proposal)
    return proposal


@router.post(
    "/{proposal_id}/activate",
    response_model=EvolutionActivationResponse,
    status_code=201,
)
async def activate_evolution_proposal(
    proposal_id: str,
    context: OrganizationContext = Depends(
        require_permission(Permission.POLICY_MANAGE)
    ),
    session: AsyncSession = Depends(get_session),
) -> EvolutionActivation:
    proposal = await _get_proposal(session, proposal_id, context.organization_id)
    _require_status(proposal, "approved")
    now = _utcnow()
    activation = EvolutionActivation(
        organization_id=context.organization_id,
        target=proposal.target,
        proposal_id=proposal.id,
        version=proposal.proposed_version,
        activated_by=context.user_id,
        activated_at=now,
    )
    proposal.status = "activated"
    proposal.activated_at = now
    proposal.updated_at = now
    session.add(activation)
    session.add(proposal)
    await session.commit()
    await session.refresh(activation)
    return activation


@router.get(
    "/active/{target:path}",
    response_model=EvolutionActivationResponse | None,
)
async def get_active_evolution(
    target: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
) -> EvolutionActivation | None:
    result = await session.execute(
        select(EvolutionActivation)
        .where(
            EvolutionActivation.organization_id == context.organization_id,
            EvolutionActivation.target == target,
            col(EvolutionActivation.rolled_back_at).is_(None),
        )
        .order_by(col(EvolutionActivation.activated_at).desc())
    )
    return result.scalars().first()


@router.post(
    "/active/{target:path}/rollback",
    response_model=EvolutionActivationResponse | None,
)
async def rollback_evolution(
    target: str,
    context: OrganizationContext = Depends(
        require_permission(Permission.POLICY_MANAGE)
    ),
    session: AsyncSession = Depends(get_session),
) -> EvolutionActivation | None:
    result = await session.execute(
        select(EvolutionActivation)
        .where(
            EvolutionActivation.organization_id == context.organization_id,
            EvolutionActivation.target == target,
            col(EvolutionActivation.rolled_back_at).is_(None),
        )
        .order_by(col(EvolutionActivation.activated_at).desc())
    )
    current = result.scalars().first()
    if current is None:
        return None
    current.rolled_back_at = _utcnow()
    proposal = await session.get(EvolutionProposal, current.proposal_id)
    if proposal is not None:
        proposal.status = "rolled_back"
        proposal.updated_at = _utcnow()
        session.add(proposal)
    session.add(current)
    await session.commit()

    restored = await session.execute(
        select(EvolutionActivation)
        .where(
            EvolutionActivation.organization_id == context.organization_id,
            EvolutionActivation.target == target,
            col(EvolutionActivation.rolled_back_at).is_(None),
        )
        .order_by(col(EvolutionActivation.activated_at).desc())
    )
    return restored.scalars().first()
