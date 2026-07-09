"""Trace storage and retrieval endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ..database import get_session
from ..models import Conversation, TraceRecord
from ..schemas import TraceListResponse, TraceResponse
from ..tenancy import OrganizationContext, Permission, require_permission

logger = logging.getLogger("aloy_backend")

router = APIRouter(prefix="/traces", tags=["traces"])


@router.get("", response_model=list[TraceListResponse])
async def list_traces(
    context: OrganizationContext = Depends(require_permission(Permission.TRACE_READ)),
    session: AsyncSession = Depends(get_session),
    limit: int = 50,
    offset: int = 0,
    conversation_id: str | None = None,
):
    """List traces (each labeled with its conversation, optionally filtered)."""
    stmt = (
        select(TraceRecord)
        .where(TraceRecord.organization_id == context.organization_id)
        .order_by(TraceRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    if conversation_id:
        stmt = stmt.where(TraceRecord.conversation_id == conversation_id)
    traces = (await session.execute(stmt)).scalars().all()

    # Resolve conversation titles in one query so the UI can group per chat.
    conv_ids = {t.conversation_id for t in traces if t.conversation_id}
    titles: dict[str, str | None] = {}
    if conv_ids:
        rows = (
            await session.execute(
                select(Conversation.id, Conversation.title).where(
                    Conversation.id.in_(conv_ids)
                )
            )
        ).all()
        titles = {cid: title for cid, title in rows}

    return [
        TraceListResponse(
            id=t.id,
            run_id=t.run_id,
            conversation_id=t.conversation_id,
            conversation_title=titles.get(t.conversation_id or ""),
            duration_seconds=t.duration_seconds,
            total_spans=t.total_spans,
            status=t.status,
            created_at=t.created_at,
        )
        for t in traces
    ]


@router.get("/{trace_id}", response_model=TraceResponse)
async def get_trace(
    trace_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.TRACE_READ)),
    session: AsyncSession = Depends(get_session),
):
    """Get a full trace with span tree."""
    trace = await session.get(TraceRecord, trace_id)
    if not trace or trace.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


@router.delete("/{trace_id}", status_code=204)
async def delete_trace(
    trace_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.ORG_MANAGE)),
    session: AsyncSession = Depends(get_session),
):
    """Delete a trace."""
    trace = await session.get(TraceRecord, trace_id)
    if not trace or trace.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Trace not found")
    await session.delete(trace)
    await session.commit()
