"""Live-run control endpoints for a conversation.

Contract: re-attach to an in-flight run's SSE stream (replay + continue),
request a cooperative stop, and resolve a paused ``ask_user`` clarification.
All state lives in ``live_runs`` / ``streaming`` — no persistence here.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ... import live_runs
from ...approvals import resolve_approval
from ...database import get_session
from ...models import ActionProposal
from ...proposal_executor import (
    ProposalDecisionError,
    decide_proposal,
    execute_proposal,
)
from ...rate_limit import rate_limited_permission
from ...streaming import resolve_clarification, subscribe_frames
from ...tenancy import OrganizationContext, Permission, require_permission
from ._helpers import _load_conv

logger = logging.getLogger("aloy_backend")

router = APIRouter()


@router.get("/{conversation_id}/live")
async def attach_live_run(
    conversation_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_READ)),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Re-attach to this conversation's in-flight run: replays every frame so
    far, then continues live — so navigating away and back resumes streaming."""
    await _load_conv(session, context, conversation_id)
    live = live_runs.get(conversation_id)
    if live is None:
        # Absence is the normal result of the app's reconnect probe, not a
        # missing resource. A 204 keeps browser consoles quiet while letting
        # the client distinguish "nothing active" from an SSE attachment.
        return Response(status_code=204)
    return StreamingResponse(
        subscribe_frames(live),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{conversation_id}/stop")
async def stop_live_run(
    conversation_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Stop this conversation's in-flight run. Cooperative: the agent halts at
    the next step boundary, then the stream finishes with a final frame (so
    every subscriber — including re-attached ones — sees a clean end)."""
    conv = await _load_conv(session, context, conversation_id)
    live = live_runs.get(conversation_id)
    if live is None or not live.request_cancel():
        raise HTTPException(status_code=404, detail="No live run")
    logger.info("Stop requested for conversation %s (run %s)", conv.id, live.run_id)
    return {"status": "stopping"}


class ClarifyBody(BaseModel):
    value: str


@router.post("/clarify/{clarification_id}")
async def submit_clarification(
    clarification_id: str,
    body: ClarifyBody,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_CREATE)
    ),
) -> dict:
    """Resolve a paused ``ask_user`` by delivering the user's answer (a tapped
    option or free text) to the waiting stream — but only if the awaiting run
    belongs to the caller (ownership enforced in resolve_clarification)."""
    if resolve_clarification(
        clarification_id,
        body.value,
        organization_id=context.organization_id,
        user_id=context.user_id,
    ):
        return {"ok": True}
    raise HTTPException(
        status_code=404, detail="Unknown or already-answered clarification"
    )


class ApprovalDecision(BaseModel):
    type: Literal["approve", "reject", "edit"]
    message: Optional[str] = None
    edited_action: Optional[dict[str, Any]] = None


class ApproveBody(BaseModel):
    decisions: list[ApprovalDecision]


@router.post("/approve/{approval_id}")
async def submit_approval(
    approval_id: str,
    body: ApproveBody,
    background_tasks: BackgroundTasks,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_CREATE)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Resolve a paused consequential-tool approval by delivering the user's
    decision (approve / reject / edit) to their waiting run — ownership enforced
    in resolve_approval so an id belonging to another user 404s."""
    if resolve_approval(
        approval_id,
        [d.model_dump() for d in body.decisions],
        organization_id=context.organization_id,
        user_id=context.user_id,
    ):
        return {"ok": True}
    if not body.decisions:
        raise HTTPException(status_code=422, detail="A decision is required")
    proposal = await session.get(ActionProposal, approval_id)
    if (
        proposal is None
        or proposal.organization_id != context.organization_id
        or proposal.user_id != context.user_id
    ):
        raise HTTPException(
            status_code=404, detail="Unknown or already-decided approval"
        )
    decision = body.decisions[0]
    try:
        proposal = await decide_proposal(
            session,
            event_id=proposal.event_id,
            proposal_id=proposal.id,
            organization_id=context.organization_id,
            user_id=context.user_id,
            actor_id=context.user_id,
            decision=decision.type,
            edited_action=decision.edited_action,
            message=decision.message,
        )
    except ProposalDecisionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if proposal.status == "approved":
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
    return {"ok": True, "proposal_id": proposal.id, "status": proposal.status}
