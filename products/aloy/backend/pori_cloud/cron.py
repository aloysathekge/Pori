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

from croniter import croniter
from sqlmodel import select

from .database import async_session
from .models import CronJob, Run

logger = logging.getLogger("pori_cloud.cron")

MIN_INTERVAL_SECONDS = 60
EVERY_PREFIX = "@every:"


def validate_schedule(schedule: str) -> None:
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
        return
    if not croniter.is_valid(raw):
        raise ValueError(f"Invalid cron expression: {raw!r}")


def compute_next_run(schedule: str, now: datetime | None = None) -> datetime:
    """Next fire time strictly after ``now`` (UTC)."""
    validate_schedule(schedule)
    moment = now or datetime.now(timezone.utc)
    raw = schedule.strip()
    if raw.startswith(EVERY_PREFIX):
        return moment + timedelta(seconds=int(raw[len(EVERY_PREFIX) :]))
    return croniter(raw, moment).get_next(datetime)


async def tick_cron_jobs(now: datetime | None = None) -> int:
    """Fire every due job once: advance its clock, enqueue its Run. Returns
    the number of runs enqueued. Safe to call from multiple workers."""
    moment = now or datetime.now(timezone.utc)
    enqueued = 0
    async with async_session() as session:
        statement = select(CronJob).where(
            CronJob.enabled == True,  # noqa: E712 - SQLAlchemy expression
            CronJob.next_run_at.is_not(None),
            CronJob.next_run_at <= moment,
        )
        if session.bind and session.bind.dialect.name == "postgresql":
            statement = statement.with_for_update(skip_locked=True)
        due_jobs = (await session.execute(statement)).scalars().all()
        for job in due_jobs:
            # Advance the clock FIRST (same transaction as the enqueue) so a
            # concurrent ticker that re-reads sees the job as not-due.
            try:
                job.next_run_at = compute_next_run(job.schedule, moment)
            except ValueError:
                logger.warning(
                    "Cron job %s has invalid schedule %r; disabling",
                    job.id,
                    job.schedule,
                )
                job.enabled = False
                session.add(job)
                continue
            run = Run(
                user_id=job.user_id,
                organization_id=job.organization_id,
                agent_id="default_agent",
                session_id="pending",
                conversation_id=job.conversation_id,
                task=job.task,
                max_steps=job.max_steps,
                status="pending",
            )
            session.add(run)
            await session.flush()
            run.session_id = run.id
            run.root_run_id = run.id
            job.last_run_at = moment
            job.last_run_id = run.id
            job.updated_at = moment
            session.add(run)
            session.add(job)
            enqueued += 1
            logger.info("Cron job %s (%s) enqueued run %s", job.id, job.name, run.id)
        await session.commit()
    return enqueued
