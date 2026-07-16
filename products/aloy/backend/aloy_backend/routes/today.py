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


def _notification_title(kind: str) -> str:
    labels = {
        "task_completed": "Task completed",
        "task_failed": "Task failed",
        "task_blocked": "Task blocked",
        "task_resumed": "Task resumed",
        "task_started": "Work started",
        "proposal_committed": "Action completed",
        "proposal_failed": "Action failed",
        "surface_published": "Surface updated",
        "surface_rolled_back": "Surface restored",
    }
    return labels.get(kind, "Event updated")


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
        return {"generated_at": now, "notifications": [], "events": []}

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

    def is_stale(task: Task) -> bool:
        updated_at = task.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        return updated_at < cutoff

    event_by_id = {event.id: event for event in events}
    stale_tasks = [
        task
        for task in tasks
        if task.status not in {"blocked", "waiting_approval"} and is_stale(task)
    ]
    represented_proposals = {proposal.id for proposal in [*pending, *committed]}
    represented_tasks = {task.id for task in stale_tasks}

    notifications: list[dict[str, Any]] = []
    for proposal in pending:
        event = event_by_id[proposal.event_id]
        notifications.append(
            {
                "id": f"proposal:{proposal.id}:pending",
                "kind": "approval_required",
                "title": "Approval requested",
                "summary": proposal.reason or proposal.impact or proposal.tool,
                "event_id": event.id,
                "event_title": event.title,
                "event_is_life": event.is_life,
                "proposal_id": proposal.id,
                "task_id": None,
                "run_id": None,
                "status": proposal.status,
                "created_at": proposal.created_at,
            }
        )
    for proposal in committed:
        event = event_by_id[proposal.event_id]
        notifications.append(
            {
                "id": f"proposal:{proposal.id}:{proposal.status}",
                "kind": "action_completed",
                "title": "Action completed",
                "summary": proposal.impact or proposal.reason or proposal.tool,
                "event_id": event.id,
                "event_title": event.title,
                "event_is_life": event.is_life,
                "proposal_id": proposal.id,
                "task_id": None,
                "run_id": None,
                "status": proposal.status,
                "created_at": proposal.updated_at,
            }
        )
    for task in stale_tasks:
        event = event_by_id[task.event_id]
        notifications.append(
            {
                "id": f"task:{task.id}:stale",
                "kind": "task_stale",
                "title": "Task needs attention",
                "summary": task.title,
                "event_id": event.id,
                "event_title": event.title,
                "event_is_life": event.is_life,
                "proposal_id": None,
                "task_id": task.id,
                "run_id": task.current_run_id,
                "status": task.status,
                "created_at": task.updated_at,
            }
        )
    for entry in activity:
        if (
            entry.proposal_id in represented_proposals
            or entry.task_id in represented_tasks
        ):
            continue
        event = event_by_id[entry.event_id]
        notifications.append(
            {
                "id": f"trail:{entry.id}",
                "kind": entry.kind,
                "title": _notification_title(entry.kind),
                "summary": entry.summary,
                "event_id": event.id,
                "event_title": event.title,
                "event_is_life": event.is_life,
                "proposal_id": entry.proposal_id,
                "task_id": entry.task_id,
                "run_id": entry.run_id,
                "status": None,
                "created_at": entry.created_at,
            }
        )
    notifications.sort(key=lambda item: item["created_at"], reverse=True)

    return {
        "generated_at": now,
        "notifications": notifications[:20],
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
                "blocked": [
                    task_payload(task)
                    for task in for_event(tasks, event.id)
                    if task.status in {"blocked", "waiting_approval"}
                ],
                "stale": [
                    task_payload(task)
                    for task in for_event(tasks, event.id)
                    if task.status not in {"blocked", "waiting_approval"}
                    and is_stale(task)
                ],
            }
            for event in events
        ],
    }


__all__ = ["get_today", "router"]
