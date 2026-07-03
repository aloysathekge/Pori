"""Durable database-leased Pori Cloud worker."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_
from sqlmodel import select

from .background import execute_claimed_run
from .config import settings
from .database import async_session
from .models import Run

logger = logging.getLogger("pori_cloud.worker")


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
    while True:
        worked = await run_once(resolved_worker_id)
        if not worked:
            await asyncio.sleep(settings.worker_poll_seconds)


def run() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    run()
