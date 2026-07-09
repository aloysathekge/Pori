"""Durable database-leased Pori Cloud worker."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_
from sqlmodel import select

from .background import execute_claimed_run
from .config import settings
from .cron import tick_cron_jobs
from .database import async_session
from .models import Run

logger = logging.getLogger("aloy_backend.worker")


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


async def claim_next_run(worker_id: str) -> str | None:
    now = datetime.now(timezone.utc)
    async with async_session() as session:
        statement = (
            select(Run)
            .where(
                Run.cancel_requested == False,
                or_(
                    Run.status == "pending",
                    (Run.status == "running")
                    & (Run.lease_expires_at.is_(None) | (Run.lease_expires_at < now)),
                ),
                Run.attempt_count < Run.max_attempts,
            )
            .order_by(Run.created_at)
            .limit(1)
        )
        if session.bind and session.bind.dialect.name == "postgresql":
            statement = statement.with_for_update(skip_locked=True)
        result = await session.execute(statement)
        run = result.scalars().first()
        if run is None:
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


def _configure_sandbox() -> None:
    """Aloy-managed isolation: point the kernel's sandbox provider at the
    configured backend once, at worker startup. When e2b is selected but its
    prereqs are missing, log and fall back to local rather than crash."""
    if not settings.sandbox_enabled:
        return
    try:
        from pori import create_sandbox_provider, set_sandbox_provider

        set_sandbox_provider(create_sandbox_provider(settings.sandbox_backend))
        logger.info("Sandbox backend active: %s", settings.sandbox_backend)
    except Exception:
        logger.exception(
            "Could not enable sandbox backend %r; agent code will run locally",
            settings.sandbox_backend,
        )


def run() -> None:
    _configure_sandbox()
    asyncio.run(serve())


if __name__ == "__main__":
    run()
