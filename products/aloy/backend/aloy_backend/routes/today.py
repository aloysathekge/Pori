"""Today: a read-only lens across the caller's Event graph."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from ..connections.store import resolve_run_connections
from ..database import get_session
from ..event_presenters import (
    event_payload,
    proposal_payload,
    task_payload,
    trail_payload,
)
from ..events import ensure_life_event
from ..models import ActionProposal, CronJob, Event, EventTrailEntry, Run, Task
from ..task_state import ACTIVE_TASK_STATUSES
from ..tenancy import OrganizationContext, Permission, require_permission
from ..tools.gmail import GmailSearchParams, gmail_search_tool

router = APIRouter(prefix="/today", tags=["today"])

_IMPORTANT_EMAIL_QUERY = (
    "in:inbox newer_than:7d {is:unread is:important} "
    "-category:promotions -category:social"
)


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
        "schedule_completed": "Scheduled work completed",
        "schedule_needs_attention": "Scheduled work needs attention",
        "schedule_failed": "Scheduled work failed",
        "schedule_cancelled": "Scheduled work cancelled",
    }
    return labels.get(kind, "Event updated")


def _email_timestamp(value: str) -> datetime | None:
    """Normalize provider dates without making Gmail metadata authoritative state."""
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@router.get("/emails")
async def get_today_emails(
    context: OrganizationContext = Depends(require_permission(Permission.RUN_READ)),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Return a bounded, provider-owned inbox brief for the Today lens.

    This is intentionally a deterministic Gmail projection: no model reads the
    messages, no Event association is guessed, and provider failure cannot make
    the rest of Today unavailable.
    """
    connections = await resolve_run_connections(
        session,
        organization_id=context.organization_id,
        user_id=context.user_id,
    )
    google = connections.get("google")
    if not google:
        return {
            "status": "not_connected",
            "account_email": None,
            "messages": [],
        }

    result = await asyncio.to_thread(
        gmail_search_tool,
        GmailSearchParams(query=_IMPORTANT_EMAIL_QUERY, max_results=5),
        {"connections": connections},
    )
    if result.get("error"):
        return {
            "status": "unavailable",
            "account_email": google.get("account_email"),
            "messages": [],
        }

    messages = []
    for message in result.get("messages", []):
        message_id = str(message.get("id") or "").strip()
        if not message_id:
            continue
        messages.append(
            {
                "id": message_id,
                "sender": str(message.get("from") or "Unknown sender")[:320],
                "subject": str(message.get("subject") or "(No subject)")[:500],
                "snippet": str(message.get("snippet") or "")[:1000],
                "received_at": _email_timestamp(str(message.get("date") or "")),
                "event_id": None,
                "event_title": None,
                "provider_url": (
                    "https://mail.google.com/mail/u/0/#all/"
                    f"{quote(message_id, safe='')}"
                ),
            }
        )

    return {
        "status": "ready",
        "account_email": google.get("account_email"),
        "messages": messages,
    }


@router.get("")
async def get_today(
    context: OrganizationContext = Depends(require_permission(Permission.RUN_READ)),
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
    scheduled_runs = (
        (
            await session.execute(
                select(Run)
                .where(
                    col(Run.event_id).in_(event_ids),
                    Run.organization_id == context.organization_id,
                    Run.user_id == context.user_id,
                    col(Run.cron_job_id).is_not(None),
                    col(Run.status).in_({"pending", "running"}),
                )
                .order_by(col(Run.created_at))
            )
        )
        .scalars()
        .all()
    )
    schedule_ids = {
        run.cron_job_id for run in scheduled_runs if run.cron_job_id is not None
    }
    schedules = (
        (
            await session.execute(
                select(CronJob).where(col(CronJob.id).in_(schedule_ids))
            )
        )
        .scalars()
        .all()
        if schedule_ids
        else []
    )
    schedule_by_id = {schedule.id: schedule for schedule in schedules}

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
        if entry.kind.startswith("schedule_"):
            # Schedule administration and start heartbeats remain in the Event
            # Trail. Today only notifies outcomes, respecting the Schedule's
            # chosen success-notification mode. Failures always surface.
            if entry.kind not in {
                "schedule_completed",
                "schedule_needs_attention",
                "schedule_failed",
                "schedule_cancelled",
            }:
                continue
            if (
                entry.kind == "schedule_completed"
                and entry.payload.get("notification_mode") != "always"
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
                "scheduled_work": [
                    {
                        "run_id": run.id,
                        "schedule_id": run.cron_job_id,
                        "schedule_name": (
                            schedule_by_id[run.cron_job_id].name
                            if run.cron_job_id in schedule_by_id
                            else "Scheduled work"
                        ),
                        "instruction": (
                            schedule_by_id[run.cron_job_id].task
                            if run.cron_job_id in schedule_by_id
                            else ""
                        ),
                        "status": run.status,
                        "created_at": run.created_at,
                        "started_at": run.started_at,
                    }
                    for run in scheduled_runs
                    if run.event_id == event.id
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


__all__ = ["get_today", "get_today_emails", "router"]
