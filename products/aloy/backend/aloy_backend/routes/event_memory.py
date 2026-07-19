"""User-facing Event memory controls."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..event_memory import (
    EventMemoryMutationError,
    correct_event_memory,
    forget_event_memory,
    list_event_memory,
    promote_event_memory,
)
from ..memory_records import record_response, row_to_record
from ..models import Event, KnowledgeEntry
from ..rate_limit import rate_limited_permission
from ..schemas import (
    EventMemoryCorrectionBody,
    EventMemoryRecordResponse,
    EventMemoryResponse,
    EventMemoryWriteResponse,
)
from ..scope_resolver import ORG
from ..tenancy import OrganizationContext, Permission

router = APIRouter(prefix="/events", tags=["event-memory"])


async def _load_event(
    session: AsyncSession,
    context: OrganizationContext,
    event_id: str,
) -> Event:
    event = await session.get(Event, event_id)
    if (
        event is None
        or event.organization_id != context.organization_id
        or event.user_id != context.user_id
    ):
        raise HTTPException(status_code=404, detail="Event not found")
    return event


def _present(
    row: KnowledgeEntry,
    *,
    scope: Literal["event", "global"],
    current_user_id: str,
    can_write: bool = True,
    active_global_ids: set[str] | None = None,
) -> EventMemoryRecordResponse:
    base = record_response(row_to_record(row)).model_dump()
    mutable = (
        can_write
        and scope == "event"
        and row.user_id == current_user_id
        and row.scope_level != ORG
    )
    promoted_global_id = (
        str((row.metadata_ or {}).get("promoted_global_id") or "") or None
    )
    if (
        promoted_global_id
        and active_global_ids is not None
        and promoted_global_id not in active_global_ids
    ):
        promoted_global_id = None
    return EventMemoryRecordResponse(
        **base,
        scope=scope,
        can_correct=mutable,
        can_forget=mutable,
        can_promote=mutable and not promoted_global_id,
        promoted_global_id=promoted_global_id,
    )


@router.get("/{event_id}/memory", response_model=EventMemoryResponse)
async def get_event_memory(
    event_id: str,
    event_limit: int = Query(default=100, ge=1, le=200),
    global_limit: int = Query(default=25, ge=0, le=100),
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.MEMORY_READ)
    ),
    session: AsyncSession = Depends(get_session),
) -> EventMemoryResponse:
    await _load_event(session, context, event_id)
    event_rows, global_rows, event_count, global_count, active_global_ids = (
        await list_event_memory(
            session,
            organization_id=context.organization_id,
            user_id=context.user_id,
            event_id=event_id,
            event_limit=event_limit,
            global_limit=global_limit,
        )
    )
    return EventMemoryResponse(
        event_id=event_id,
        event_records=[
            _present(
                row,
                scope="event",
                current_user_id=context.user_id,
                can_write=context.permits(Permission.MEMORY_WRITE),
                active_global_ids=active_global_ids,
            )
            for row in event_rows
        ],
        inherited_global_records=[
            _present(
                row,
                scope="global",
                current_user_id=context.user_id,
                can_write=context.permits(Permission.MEMORY_WRITE),
            )
            for row in global_rows
        ],
        event_count=event_count,
        inherited_global_count=global_count,
    )


@router.post(
    "/{event_id}/memory/{memory_id}/corrections",
    response_model=EventMemoryWriteResponse,
)
async def correct_memory(
    event_id: str,
    memory_id: str,
    body: EventMemoryCorrectionBody,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.MEMORY_WRITE)
    ),
    session: AsyncSession = Depends(get_session),
) -> EventMemoryWriteResponse:
    event = await _load_event(session, context, event_id)
    try:
        row = await correct_event_memory(
            session,
            event=event,
            organization_id=context.organization_id,
            user_id=context.user_id,
            memory_id=memory_id,
            content=body.content,
            reason=body.reason,
        )
    except EventMemoryMutationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    await session.commit()
    return EventMemoryWriteResponse(
        record=_present(row, scope="event", current_user_id=context.user_id),
        created=True,
    )


@router.delete("/{event_id}/memory/{memory_id}", status_code=204)
async def forget_memory(
    event_id: str,
    memory_id: str,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.MEMORY_WRITE)
    ),
    session: AsyncSession = Depends(get_session),
) -> None:
    event = await _load_event(session, context, event_id)
    try:
        await forget_event_memory(
            session,
            event=event,
            organization_id=context.organization_id,
            user_id=context.user_id,
            memory_id=memory_id,
        )
    except EventMemoryMutationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    await session.commit()


@router.post(
    "/{event_id}/memory/{memory_id}/promote",
    response_model=EventMemoryWriteResponse,
)
async def promote_memory(
    event_id: str,
    memory_id: str,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.MEMORY_WRITE)
    ),
    session: AsyncSession = Depends(get_session),
) -> EventMemoryWriteResponse:
    event = await _load_event(session, context, event_id)
    try:
        row, created = await promote_event_memory(
            session,
            event=event,
            organization_id=context.organization_id,
            user_id=context.user_id,
            memory_id=memory_id,
        )
    except EventMemoryMutationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    await session.commit()
    return EventMemoryWriteResponse(
        record=_present(row, scope="global", current_user_id=context.user_id),
        created=created,
    )
