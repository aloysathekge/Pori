"""Durable database-leased Pori Cloud worker."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, or_
from sqlmodel import col, select

from .background import execute_claimed_run
from .config import settings
from .cron import tick_cron_jobs
from .database import async_session
from .models import Event, Organization, Run
from .proposal_executor import (
    execute_next_approved_proposal,
    expire_due_proposals,
    reconcile_stale_executions,
)
from .tenancy import OrganizationPolicy

logger = logging.getLogger("aloy_backend.worker")


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


async def claim_next_run(worker_id: str) -> str | None:
    now = datetime.now(timezone.utc)
    async with async_session() as session:
        candidate_statement = (
            select(Run.id)
            .where(
                Run.cancel_requested == False,
                or_(
                    col(Run.status) == "pending",
                    (col(Run.status) == "running")
                    & (
                        col(Run.lease_expires_at).is_(None)
                        | (col(Run.lease_expires_at) < now)
                    ),
                ),
                Run.attempt_count < Run.max_attempts,
            )
            .order_by(col(Run.created_at))
            .limit(50)
        )
        candidate_ids = list(
            (await session.execute(candidate_statement)).scalars().all()
        )
        run: Run | None = None
        dialect = session.bind.dialect.name if session.bind else ""
        for candidate_id in candidate_ids:
            lock_statement = select(Run).where(Run.id == candidate_id)
            if dialect == "postgresql":
                lock_statement = lock_statement.with_for_update(skip_locked=True)
            candidate = (await session.execute(lock_statement)).scalars().first()
            if candidate is None or candidate.cancel_requested:
                continue
            candidate_lease = candidate.lease_expires_at
            if candidate_lease is not None and candidate_lease.tzinfo is None:
                candidate_lease = candidate_lease.replace(tzinfo=timezone.utc)
            if not (
                candidate.status == "pending"
                or (
                    candidate.status == "running"
                    and (candidate_lease is None or candidate_lease < now)
                )
            ):
                continue

            organization_statement = select(Organization).where(
                Organization.id == candidate.organization_id
            )
            if dialect == "postgresql":
                # Serialize admission for the account. This makes the cap and
                # Event/Conversation checks safe with several worker processes.
                organization_statement = organization_statement.with_for_update()
            organization = (
                (await session.execute(organization_statement)).scalars().first()
            )
            # Legacy/imported Run rows can predate the Organization row. Claim
            # them under the safe default cap so execute_claimed_run can apply
            # its authoritative membership check and fail them durably.
            policy = (
                OrganizationPolicy.model_validate(organization.policy or {})
                if organization is not None
                else OrganizationPolicy()
            )
            account_cap = min(
                policy.max_concurrent_runs,
                max(1, settings.max_concurrent_runs),
            )
            active_lease = or_(
                col(Run.lease_expires_at).is_(None),
                col(Run.lease_expires_at) >= now,
            )
            active_account = (
                await session.execute(
                    select(func.count())
                    .select_from(Run)
                    .where(
                        Run.organization_id == candidate.organization_id,
                        Run.user_id == candidate.user_id,
                        Run.id != candidate.id,
                        Run.status == "running",
                        active_lease,
                    )
                )
            ).scalar_one()
            if active_account >= account_cap:
                continue

            if candidate.conversation_id:
                active_conversation = (
                    await session.execute(
                        select(func.count())
                        .select_from(Run)
                        .where(
                            Run.organization_id == candidate.organization_id,
                            Run.conversation_id == candidate.conversation_id,
                            Run.id != candidate.id,
                            Run.status == "running",
                            active_lease,
                        )
                    )
                ).scalar_one()
                if active_conversation:
                    continue

            if candidate.task_id:
                event_statement = select(Event).where(Event.id == candidate.event_id)
                if dialect == "postgresql":
                    event_statement = event_statement.with_for_update()
                if (await session.execute(event_statement)).scalars().first() is None:
                    continue
                active_event_task = (
                    await session.execute(
                        select(func.count())
                        .select_from(Run)
                        .where(
                            Run.organization_id == candidate.organization_id,
                            Run.event_id == candidate.event_id,
                            col(Run.task_id).is_not(None),
                            Run.id != candidate.id,
                            Run.status == "running",
                            active_lease,
                        )
                    )
                ).scalar_one()
                if active_event_task:
                    continue
            run = candidate
            break

        if run is None:
            await session.rollback()
            return None
        run.status = "running"
        run.attempt_count += 1
        run.lease_owner = worker_id
        lease_seconds = max(settings.worker_lease_seconds, run.timeout_seconds + 30)
        run.lease_expires_at = now + timedelta(seconds=lease_seconds)
        run.started_at = run.started_at or now
        session.add(run)
        await session.commit()
        return run.id


async def run_once(worker_id: str | None = None) -> bool:
    resolved_worker_id = worker_id or default_worker_id()
    await expire_due_proposals()
    await reconcile_stale_executions()
    proposal_result = await execute_next_approved_proposal()
    if proposal_result is not None:
        return True
    run_id = await claim_next_run(resolved_worker_id)
    if run_id is None:
        return False
    await execute_claimed_run(run_id, resolved_worker_id)
    return True


async def serve(worker_id: str | None = None) -> None:
    resolved_worker_id = worker_id or default_worker_id()
    logger.info("Pori Cloud worker started: %s", resolved_worker_id)
    # Cron piggybacks on the worker loop: every cron_tick_seconds, due jobs
    # are advanced-then-enqueued (at-most-once even with several workers —
    # see cron.tick_cron_jobs). No separate scheduler process to deploy.
    last_cron_tick = 0.0
    while True:
        if time.monotonic() - last_cron_tick >= settings.cron_tick_seconds:
            last_cron_tick = time.monotonic()
            try:
                await tick_cron_jobs()
            except Exception:
                logger.exception("Cron tick failed; will retry next tick")
        # One bad run must never take down the loop (execute_claimed_run guards
        # its own persistence, but claim/DB errors can still surface here).
        try:
            worked = await run_once(resolved_worker_id)
        except Exception:
            logger.exception("run_once failed; continuing")
            worked = False
        if not worked:
            await asyncio.sleep(settings.worker_poll_seconds)


def run() -> None:
    from .orchestrator import configure_sandbox

    configure_sandbox()
    asyncio.run(serve())


if __name__ == "__main__":
    run()
