"""Today: a read-only lens across the caller's Event graph."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from ..database import get_session
from ..event_presenters import (
    event_payload,
    proposal_payload,
    task_payload,
    trail_payload,
)
from ..events import ensure_life_event
from ..models import ActionProposal, Event, EventTrailEntry, Task
from ..rate_limit import rate_limited_permission
from ..task_state import ACTIVE_TASK_STATUSES
from ..tenancy import OrganizationContext, Permission

router = APIRouter(prefix="/today", tags=["today"])


@router.get("")
async def get_today(
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_READ)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Aggregate attention, recent evidence, and open work by Event."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)
    await ensure_life_event(
        session,
        organization_id=context.organization_id,
        user_id=context.user_id,
    )
    events = (
        (
            await session.execute(
                select(Event)
                .where(
                    Event.organization_id == context.organization_id,
                    Event.user_id == context.user_id,
                    Event.lifecycle != "archived",
                )
                .order_by(col(Event.is_life).desc(), col(Event.updated_at).desc())
            )
        )
        .scalars()
        .all()
    )
    event_ids = [event.id for event in events]
    if not event_ids:
        return {"generated_at": now, "events": []}

    pending = (
        (
            await session.execute(
                select(ActionProposal)
                .where(
                    col(ActionProposal.event_id).in_(event_ids),
                    ActionProposal.organization_id == context.organization_id,
                    ActionProposal.user_id == context.user_id,
                    ActionProposal.status == "pending",
                )
                .order_by(col(ActionProposal.created_at).desc())
            )
        )
        .scalars()
        .all()
    )
    committed = (
        (
            await session.execute(
                select(ActionProposal)
                .where(
                    col(ActionProposal.event_id).in_(event_ids),
                    ActionProposal.organization_id == context.organization_id,
                    ActionProposal.user_id == context.user_id,
                    ActionProposal.status == "committed",
                    ActionProposal.updated_at >= cutoff,
                )
                .order_by(col(ActionProposal.updated_at).desc())
            )
        )
        .scalars()
        .all()
    )
    activity = (
        (
            await session.execute(
                select(EventTrailEntry)
                .where(
                    col(EventTrailEntry.event_id).in_(event_ids),
                    EventTrailEntry.organization_id == context.organization_id,
                    EventTrailEntry.user_id == context.user_id,
                    EventTrailEntry.created_at >= cutoff,
                )
                .order_by(col(EventTrailEntry.created_at).desc())
            )
        )
        .scalars()
        .all()
    )
    tasks = (
        (
            await session.execute(
                select(Task)
                .where(
                    col(Task.event_id).in_(event_ids),
                    Task.organization_id == context.organization_id,
                    Task.user_id == context.user_id,
                    col(Task.status).in_(ACTIVE_TASK_STATUSES),
                )
                .order_by(col(Task.order), col(Task.created_at))
            )
        )
        .scalars()
        .all()
    )

    def for_event(rows, event_id: str) -> list:
        return [row for row in rows if row.event_id == event_id]

    return {
        "generated_at": now,
        "events": [
            {
                "event": event_payload(event),
                "needs_decision": [
                    proposal_payload(proposal)
                    for proposal in for_event(pending, event.id)
                ],
                "changed_proposals": [
                    proposal_payload(proposal)
                    for proposal in for_event(committed, event.id)
                ],
                "activity": [
                    trail_payload(entry) for entry in for_event(activity, event.id)
                ],
                "upcoming": [task_payload(task) for task in for_event(tasks, event.id)],
            }
            for event in events
        ],
    }


__all__ = ["get_today", "router"]
