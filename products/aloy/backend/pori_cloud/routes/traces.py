"""Trace storage and retrieval endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ..database import get_session
from ..models import TraceRecord
from ..schemas import TraceListResponse, TraceResponse
from ..tenancy import OrganizationContext, Permission, require_permission

logger = logging.getLogger("pori_cloud")

router = APIRouter(prefix="/traces", tags=["traces"])


@router.get("", response_model=list[TraceListResponse])
async def list_traces(
    context: OrganizationContext = Depends(require_permission(Permission.TRACE_READ)),
    session: AsyncSession = Depends(get_session),
    limit: int = 50,
    offset: int = 0,
):
    """List traces for the current user (without full trace data)."""
    result = await session.execute(
        select(TraceRecord)
        .where(TraceRecord.organization_id == context.organization_id)
        .order_by(TraceRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return result.scalars().all()


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
