"""Event-owned Proposal decision endpoints."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..database import get_session
from ..proposal_executor import (
    ProposalDecisionError,
    decide_proposal,
    execute_proposal,
)
from ..rate_limit import rate_limited_permission
from ..tenancy import OrganizationContext, Permission

router = APIRouter(prefix="/events", tags=["events"])


class ProposalDecisionBody(BaseModel):
    decision: Literal["approve", "reject", "edit"]
    message: str | None = None
    edited_action: dict[str, Any] | None = None


@router.post("/{event_id}/proposals/{proposal_id}/decision")
async def submit_proposal_decision(
    event_id: str,
    proposal_id: str,
    body: ProposalDecisionBody,
    background_tasks: BackgroundTasks,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_CREATE)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Persist a decision; approved Proposals execute on the same dumb rail."""
    try:
        proposal = await decide_proposal(
            session,
            event_id=event_id,
            proposal_id=proposal_id,
            organization_id=context.organization_id,
            user_id=context.user_id,
            actor_id=context.user_id,
            decision=body.decision,
            edited_action=body.edited_action,
            message=body.message,
        )
    except ProposalDecisionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    if proposal.status == "approved":
        # Use a fresh session bound to the request's engine. This works for the
        # production engine and for dependency-overridden test engines.
        session_factory = async_sessionmaker(
            session.bind,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        background_tasks.add_task(
            execute_proposal,
            proposal.id,
            session_factory=session_factory,
        )
    return {
        "proposal_id": proposal.id,
        "status": proposal.status,
        "decision": body.decision,
    }


__all__ = ["ProposalDecisionBody", "router", "submit_proposal_decision"]
