"""Live-run control endpoints for a conversation.

Contract: re-attach to an in-flight run's SSE stream (replay + continue),
request a cooperative stop, and resolve a paused ``ask_user`` clarification.
All state lives in ``live_runs`` / ``streaming`` — no persistence here.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ... import live_runs
from ...database import get_session
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
):
    """Re-attach to this conversation's in-flight run: replays every frame so
    far, then continues live — so navigating away and back resumes streaming."""
    await _load_conv(session, context, conversation_id)
    live = live_runs.get(conversation_id)
    if live is None:
        raise HTTPException(status_code=404, detail="No live run")
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
):
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
):
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
