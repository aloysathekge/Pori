"""Frozen authority and lifecycle evidence for Event-owned Schedule runs."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from .models import Event, EventTrailEntry, Run
from .tools import GOOGLE_WRITE_TOOLS

SCHEDULE_AUTHORITIES = frozenset({"report_only", "organize"})
SCHEDULE_NOTIFICATION_MODES = frozenset({"attention", "always"})

# Unattended runs never evolve Aloy or commission a generated application.
# Surface state remains readable, but a specialist build must be requested in
# an attended Event conversation. MCP is separately withheld in background.py
# until MCP tools carry host-verifiable read/write authority metadata.
_ALWAYS_DENIED = frozenset(
    {
        "request_event_surface",
        "propose_evolution",
        "write_skill",
    }
)
_REPORT_ONLY_DENIED = frozenset(
    {
        "task_create",
        "task_update",
        "gmail_create_draft",
        "remember",
        "archival_memory_insert",
        "core_memory_append",
        "core_memory_replace",
        "memory_insert",
        "memory_rethink",
        "core_memory_rethink",
        "write_file",
        "create_directory",
        "copy_file",
        "move_file",
        "delete_file",
        "edit_file",
        *_ALWAYS_DENIED,
        *GOOGLE_WRITE_TOOLS,
    }
)


def frozen_schedule_profile(
    *,
    schedule_id: str,
    schedule_name: str,
    authority: str,
    notification_mode: str,
    timezone_name: str,
    scheduled_for: str,
    next_run_at: str,
) -> dict:
    """Credential-free immutable authority captured when a run is enqueued."""
    return {
        "profile_id": "aloy.event-schedule",
        "version": 1,
        "schedule_id": schedule_id,
        "schedule_name": schedule_name,
        "authority": authority,
        "notification_mode": notification_mode,
        "timezone": timezone_name,
        "scheduled_for": scheduled_for,
        "next_run_at": next_run_at,
    }


def schedule_authority(run: Run) -> str | None:
    if not run.cron_job_id:
        return None
    profile = run.run_profile or {}
    authority = str(profile.get("authority") or "report_only")
    return authority if authority in SCHEDULE_AUTHORITIES else "report_only"


def scheduled_denied_tools(run: Run) -> frozenset[str]:
    authority = schedule_authority(run)
    if authority is None:
        return frozenset()
    return _REPORT_ONLY_DENIED if authority == "report_only" else _ALWAYS_DENIED


def scheduled_instruction(*, name: str, instruction: str, authority: str) -> str:
    authority_rule = (
        "Read and report only. Do not create or update Event Tasks or files, create "
        "drafts, request a Surface, or change external state."
        if authority == "report_only"
        else "You may organize durable state inside this Event and prepare reversible "
        "drafts. Any consequential external action must remain a Proposal awaiting "
        "the user's approval."
    )
    return (
        "A durable Event Schedule woke you without a new user message.\n"
        f"Schedule: {name}\n"
        f"Authority: {authority_rule}\n"
        "Work only within the current Event and its trusted context. State clearly "
        "what changed, what needs attention, and what should happen next. Never claim "
        "an external action completed without a committed receipt.\n\n"
        f"Scheduled instruction:\n{instruction}"
    )


async def record_schedule_terminal_trail(
    session: AsyncSession, *, run: Run
) -> EventTrailEntry | None:
    """Record one terminal Trail entry for a Schedule occurrence."""
    if not run.cron_job_id or run.status not in {
        "completed",
        "failed",
        "cancelled",
    }:
        return None
    existing = (
        (
            await session.execute(
                select(EventTrailEntry).where(
                    EventTrailEntry.event_id == run.event_id,
                    EventTrailEntry.run_id == run.id,
                    col(EventTrailEntry.kind).in_(
                        {
                            "schedule_completed",
                            "schedule_needs_attention",
                            "schedule_failed",
                            "schedule_cancelled",
                        }
                    ),
                )
            )
        )
        .scalars()
        .first()
    )
    if existing is not None:
        return existing

    profile = run.run_profile or {}
    name = str(profile.get("schedule_name") or "Scheduled work")[:120]
    if run.status == "cancelled":
        kind = "schedule_cancelled"
        summary = f"Cancelled scheduled run: {name}"
    elif run.status == "failed":
        kind = "schedule_failed"
        summary = f"Scheduled run failed: {name}"
    elif run.success:
        kind = "schedule_completed"
        summary = f"Completed scheduled run: {name}"
    else:
        kind = "schedule_needs_attention"
        summary = f"Scheduled run needs attention: {name}"

    entry = EventTrailEntry(
        organization_id=run.organization_id,
        user_id=run.user_id,
        event_id=run.event_id,
        actor_id=run.agent_id,
        kind=kind,
        summary=summary,
        run_id=run.id,
        payload={
            "schedule_id": run.cron_job_id,
            "schedule_name": name,
            "notification_mode": str(profile.get("notification_mode") or "attention"),
            "status": run.status,
            "success": run.success,
            "steps_taken": run.steps_taken,
        },
    )
    session.add(entry)
    event = await session.get(Event, run.event_id)
    if event is not None:
        event.updated_at = datetime.now(timezone.utc)
        session.add(event)
    return entry


__all__ = [
    "SCHEDULE_AUTHORITIES",
    "SCHEDULE_NOTIFICATION_MODES",
    "frozen_schedule_profile",
    "record_schedule_terminal_trail",
    "schedule_authority",
    "scheduled_denied_tools",
    "scheduled_instruction",
]
