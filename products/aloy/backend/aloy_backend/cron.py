"""Cron engine: recurring tasks that enqueue Runs on the worker queue.

Harvested pattern (Hermes cron, docs/long-running.md Phase 3): per tick,
advance ``next_run_at`` for every due job INSIDE the same transaction that
enqueues its Run, before commit — so a job fires at most once per due time
even with several workers ticking concurrently (row locks on Postgres via
``skip_locked``; SQLite's single-writer semantics cover dev).

Schedules are either a 5-field cron expression ("0 7 * * 1-5", parsed by
croniter) or the plain-interval form ``@every:SECONDS``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
from sqlmodel import col, select

from .database import async_session
from .events import ensure_event_conversation
from .models import CronJob, Event, EventTrailEntry, Organization, Run
from .run_budgets import resolve_run_budget
from .schedule_runtime import (
    SCHEDULE_AUTHORITIES,
    SCHEDULE_NOTIFICATION_MODES,
    frozen_schedule_profile,
    scheduled_instruction,
)
from .tenancy import OrganizationPolicy

logger = logging.getLogger("aloy_backend.cron")

MIN_INTERVAL_SECONDS = 60
EVERY_PREFIX = "@every:"


def validate_timezone(timezone_name: str) -> None:
    """Raise ValueError when an IANA timezone cannot be resolved."""
    raw = (timezone_name or "").strip()
    if not raw:
        raise ValueError("timezone is required")
    try:
        ZoneInfo(raw)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"Unknown IANA timezone: {raw!r}") from None


def validate_schedule(schedule: str, timezone_name: str = "UTC") -> None:
    """Raise ValueError if the schedule is not a supported expression."""
    raw = (schedule or "").strip()
    if not raw:
        raise ValueError("schedule is required")
    if raw.startswith(EVERY_PREFIX):
        try:
            seconds = int(raw[len(EVERY_PREFIX) :])
        except ValueError:
            raise ValueError(
                f"Invalid @every schedule {raw!r}: expected @every:SECONDS"
            ) from None
        if seconds < MIN_INTERVAL_SECONDS:
            raise ValueError(
                f"@every interval must be >= {MIN_INTERVAL_SECONDS} seconds"
            )
        validate_timezone(timezone_name)
        return
    if not croniter.is_valid(raw):
        raise ValueError(f"Invalid cron expression: {raw!r}")
    validate_timezone(timezone_name)


def compute_next_run(
    schedule: str,
    now: datetime | None = None,
    timezone_name: str = "UTC",
) -> datetime:
    """Next fire time strictly after ``now`` (UTC)."""
    validate_schedule(schedule, timezone_name)
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    raw = schedule.strip()
    if raw.startswith(EVERY_PREFIX):
        return moment + timedelta(seconds=int(raw[len(EVERY_PREFIX) :]))
    local_timezone = ZoneInfo(timezone_name)
    local_now = moment.astimezone(local_timezone)
    next_local = croniter(raw, local_now).get_next(datetime)
    if next_local.tzinfo is None:
        next_local = next_local.replace(tzinfo=local_timezone)
    return next_local.astimezone(timezone.utc)


async def tick_cron_jobs(now: datetime | None = None) -> int:
    """Fire every due job once: advance its clock, enqueue its Run. Returns
    the number of runs enqueued. Safe to call from multiple workers."""
    moment = now or datetime.now(timezone.utc)
    enqueued = 0
    async with async_session() as session:
        statement = select(CronJob).where(
            CronJob.enabled == True,  # noqa: E712 - SQLAlchemy expression
            col(CronJob.next_run_at).is_not(None),
            col(CronJob.next_run_at) <= moment,
        )
        if session.bind and session.bind.dialect.name == "postgresql":
            statement = statement.with_for_update(skip_locked=True)
        due_jobs = (await session.execute(statement)).scalars().all()
        for job in due_jobs:
            # Advance the clock FIRST (same transaction as the enqueue) so a
            # concurrent ticker that re-reads sees the job as not-due.
            try:
                scheduled_for = job.next_run_at or moment
                job.next_run_at = compute_next_run(job.schedule, moment, job.timezone)
            except ValueError:
                logger.warning(
                    "Cron job %s has invalid schedule %r; disabling",
                    job.id,
                    job.schedule,
                )
                job.enabled = False
                session.add(job)
                continue
            event = await session.get(Event, job.event_id) if job.event_id else None
            if event is None:
                # A legacy row without an Event cannot safely guess its scope.
                # Preserve it for inspection, but stop unattended execution.
                job.enabled = False
                session.add(job)
                logger.warning("Cron job %s has no Event; disabling", job.id)
                continue
            if (
                event.organization_id != job.organization_id
                or event.user_id != job.user_id
            ):
                job.enabled = False
                session.add(job)
                logger.warning(
                    "Cron job %s has invalid Event ownership; disabling", job.id
                )
                continue
            if event.lifecycle != "active":
                # Dormant/archived Events stay quiet. Advance past the missed
                # occurrence so reactivation does not replay a backlog.
                session.add(job)
                continue

            conversation = await ensure_event_conversation(session, event=event)
            job.conversation_id = conversation.id
            authority = (
                job.authority
                if job.authority in SCHEDULE_AUTHORITIES
                else "report_only"
            )
            notification_mode = (
                job.notification_mode
                if job.notification_mode in SCHEDULE_NOTIFICATION_MODES
                else "attention"
            )
            run_profile = frozen_schedule_profile(
                schedule_id=job.id,
                schedule_name=job.name,
                authority=authority,
                notification_mode=notification_mode,
                timezone_name=job.timezone,
                scheduled_for=scheduled_for.isoformat(),
                next_run_at=job.next_run_at.isoformat(),
            )
            organization = await session.get(Organization, job.organization_id)
            policy = (
                OrganizationPolicy.model_validate(organization.policy or {})
                if organization is not None
                else OrganizationPolicy()
            )
            budget = resolve_run_budget(
                policy,
                {"max_steps": job.max_steps},
            )
            run = Run(
                user_id=job.user_id,
                organization_id=job.organization_id,
                event_id=event.id,
                agent_id="default_agent",
                session_id="pending",
                conversation_id=conversation.id,
                cron_job_id=job.id,
                run_kind="scheduled",
                run_profile=run_profile,
                task=scheduled_instruction(
                    name=job.name,
                    instruction=job.task,
                    authority=authority,
                ),
                max_steps=budget.max_steps,
                max_tool_calls=budget.max_tool_calls,
                max_tokens=budget.max_tokens,
                max_cost_usd=budget.max_cost_usd,
                timeout_seconds=budget.timeout_seconds,
                max_attempts=policy.max_attempts,
                status="pending",
            )
            session.add(run)
            await session.flush()
            run.session_id = run.id
            run.root_run_id = run.id
            job.last_run_at = moment
            job.last_run_id = run.id
            job.updated_at = moment
            event.updated_at = moment
            session.add(event)
            session.add(run)
            session.add(job)
            session.add(
                EventTrailEntry(
                    organization_id=job.organization_id,
                    user_id=job.user_id,
                    event_id=event.id,
                    actor_id="aloy:scheduler",
                    kind="schedule_triggered",
                    summary=f"Started scheduled run: {job.name}",
                    run_id=run.id,
                    payload={
                        "schedule_id": job.id,
                        "schedule_name": job.name,
                        "wake_reason": "time_trigger",
                        "scheduled_for": scheduled_for.isoformat(),
                        "next_run_at": job.next_run_at.isoformat(),
                        "timezone": job.timezone,
                        "authority": authority,
                        "notification_mode": notification_mode,
                    },
                )
            )
            enqueued += 1
            logger.info("Cron job %s (%s) enqueued run %s", job.id, job.name, run.id)
        await session.commit()
    return enqueued


__all__ = [
    "compute_next_run",
    "tick_cron_jobs",
    "validate_schedule",
    "validate_timezone",
]
