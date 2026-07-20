"""Bounded recovery for Runs and Tasks that can no longer make progress."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from .database import async_session
from .models import Conversation, Event, EventTrailEntry, Run, Task
from .schedule_runtime import record_schedule_terminal_trail
from .surface_lifecycle import reconcile_surface_run
from .task_execution import add_task_lifecycle_message, task_has_pending_proposal
from .task_state import mutate_task

WATCHDOG_BATCH_SIZE = 100
ACTIVE_RUN_STATUSES = frozenset({"pending", "running"})
ACTIVE_TASK_STATUSES = frozenset({"queued", "in_progress"})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _lease_is_expired(run: Run, now: datetime) -> bool:
    lease = _as_utc(run.lease_expires_at)
    return run.status == "running" and (lease is None or lease < now)


def _watchdog_reason(run: Run, terminal_status: str) -> tuple[str, str]:
    if terminal_status == "cancelled":
        return (
            "cancel_requested_after_worker_loss",
            "The interrupted Run was stopped after its worker lease expired.",
        )
    return (
        "worker_lease_expired_attempts_exhausted",
        "The Run stopped because its worker lease expired after all safe retry "
        "attempts were used.",
    )


async def _task_conversation(
    session: AsyncSession,
    *,
    run: Run,
    event: Event,
    task: Task,
) -> Conversation | None:
    conversation_id = (
        run.conversation_id
        or task.origin_conversation_id
        or event.primary_conversation_id
    )
    if not conversation_id:
        return None
    conversation = await session.get(Conversation, conversation_id)
    if (
        conversation is None
        or conversation.event_id != run.event_id
        or conversation.organization_id != run.organization_id
        or conversation.user_id != run.user_id
    ):
        return None
    return conversation


async def _reconcile_active_task(
    session: AsyncSession,
    *,
    run: Run,
    terminal_status: str,
    reason: str,
) -> Task | None:
    if not run.task_id:
        return None
    task = await session.get(Task, run.task_id, populate_existing=True)
    event = await session.get(Event, run.event_id, populate_existing=True)
    if (
        task is None
        or event is None
        or task.current_run_id != run.id
        or task.organization_id != run.organization_id
        or task.user_id != run.user_id
        or task.status not in ACTIVE_TASK_STATUSES
    ):
        return task

    target_status = "cancelled" if terminal_status == "cancelled" else "failed"
    changes: dict[str, Any] = {"status": target_status, "blocker": ""}
    if target_status == "failed":
        changes["result_summary"] = reason
    await mutate_task(
        session,
        event=event,
        task=task,
        changes=changes,
        actor_id="worker:run-watchdog",
        source_run_id=run.id,
    )
    conversation = await _task_conversation(
        session,
        run=run,
        event=event,
        task=task,
    )
    if conversation is not None:
        content = (
            f"Stopped **{task.title}** after its interrupted Run was cancelled."
            if target_status == "cancelled"
            else (
                f"**{task.title}** could not recover after its worker stopped. "
                "You can retry it safely."
            )
        )
        add_task_lifecycle_message(
            session,
            conversation_id=conversation.id,
            task=task,
            run_id=run.id,
            status=target_status,
            content=content,
        )
        conversation.updated_at = _utcnow()
        session.add(conversation)
    return task


async def reconcile_stale_runs(
    *,
    session_factory: Any = async_session,
    now: datetime | None = None,
    batch_size: int = WATCHDOG_BATCH_SIZE,
) -> int:
    """Terminalize expired Runs that are unsafe or impossible to reclaim.

    Recoverable expired Runs remain untouched for ``claim_next_run`` to resume
    from their durable checkpoint. A cancellation request or exhausted final
    attempt is terminalized exactly once under a row lock.
    """
    resolved_now = now or _utcnow()
    expired_lease = or_(
        col(Run.lease_expires_at).is_(None),
        col(Run.lease_expires_at) < resolved_now,
    )
    terminal_candidate = or_(
        col(Run.cancel_requested).is_(True),
        col(Run.attempt_count) >= col(Run.max_attempts),
    )
    async with session_factory() as discovery:
        candidate_ids = list(
            (
                await discovery.execute(
                    select(Run.id)
                    .where(
                        col(Run.status).in_(ACTIVE_RUN_STATUSES),
                        terminal_candidate,
                        or_(
                            col(Run.status) == "pending",
                            and_(col(Run.status) == "running", expired_lease),
                        ),
                    )
                    .order_by(col(Run.created_at))
                    .limit(max(1, batch_size))
                )
            )
            .scalars()
            .all()
        )

    reconciled = 0
    for run_id in candidate_ids:
        async with session_factory() as session:
            statement = select(Run).where(Run.id == run_id)
            if session.bind is not None and session.bind.dialect.name == "postgresql":
                statement = statement.with_for_update(skip_locked=True)
            run = (await session.execute(statement)).scalars().first()
            if run is None or run.status not in ACTIVE_RUN_STATUSES:
                continue
            if run.status == "running" and not _lease_is_expired(run, resolved_now):
                continue
            if not run.cancel_requested and run.attempt_count < run.max_attempts:
                continue

            terminal_status = "cancelled" if run.cancel_requested else "failed"
            code, reason = _watchdog_reason(run, terminal_status)
            previous_status = run.status
            previous_lease_owner = run.lease_owner
            previous_lease_expires_at = _as_utc(run.lease_expires_at)
            run.status = terminal_status
            run.success = False
            run.completed_at = resolved_now
            run.lease_owner = None
            run.lease_expires_at = None
            run.progress = {
                **(run.progress or {}),
                "watchdog": {
                    "code": code,
                    "at": resolved_now.isoformat(),
                    "attempt_count": run.attempt_count,
                    "max_attempts": run.max_attempts,
                },
            }
            session.add(run)
            await _reconcile_active_task(
                session,
                run=run,
                terminal_status=terminal_status,
                reason=reason,
            )
            session.add(
                EventTrailEntry(
                    organization_id=run.organization_id,
                    user_id=run.user_id,
                    event_id=run.event_id,
                    actor_id="worker:run-watchdog",
                    kind="run_watchdog_terminal",
                    summary=(
                        "Cancelled an interrupted Run"
                        if terminal_status == "cancelled"
                        else "Marked an unrecoverable Run as failed"
                    ),
                    run_id=run.id,
                    task_id=run.task_id,
                    payload={
                        "code": code,
                        "before": {
                            "status": previous_status,
                            "lease_owner": previous_lease_owner,
                            "lease_expires_at": (
                                previous_lease_expires_at.isoformat()
                                if previous_lease_expires_at
                                else None
                            ),
                        },
                        "after": {"status": terminal_status},
                        "attempt_count": run.attempt_count,
                        "max_attempts": run.max_attempts,
                    },
                )
            )
            await reconcile_surface_run(session, run=run, error=reason)
            await record_schedule_terminal_trail(session, run=run)
            await session.commit()
            reconciled += 1
    return reconciled


async def reconcile_orphaned_tasks(
    *,
    session_factory: Any = async_session,
    batch_size: int = WATCHDOG_BATCH_SIZE,
) -> int:
    """Repair Tasks whose recorded Run is missing, mismatched, or terminal."""
    async with session_factory() as discovery:
        invalid_or_terminal_run = or_(
            col(Task.current_run_id).is_(None),
            col(Run.id).is_(None),
            col(Run.status).not_in(ACTIVE_RUN_STATUSES),
            col(Run.task_id).is_(None),
            col(Run.task_id) != col(Task.id),
            col(Run.event_id) != col(Task.event_id),
            col(Run.organization_id) != col(Task.organization_id),
            col(Run.user_id) != col(Task.user_id),
        )
        task_ids = list(
            (
                await discovery.execute(
                    select(Task.id)
                    .outerjoin(Run, col(Task.current_run_id) == col(Run.id))
                    .where(
                        col(Task.status).in_(ACTIVE_TASK_STATUSES),
                        invalid_or_terminal_run,
                    )
                    .order_by(col(Task.updated_at))
                    .limit(max(1, batch_size))
                )
            )
            .scalars()
            .all()
        )

    reconciled = 0
    for task_id in task_ids:
        async with session_factory() as session:
            statement = select(Task).where(Task.id == task_id)
            if session.bind is not None and session.bind.dialect.name == "postgresql":
                statement = statement.with_for_update(skip_locked=True)
            task = (await session.execute(statement)).scalars().first()
            if task is None or task.status not in ACTIVE_TASK_STATUSES:
                continue
            run = (
                await session.get(Run, task.current_run_id)
                if task.current_run_id
                else None
            )
            run_matches = bool(
                run is not None
                and run.task_id == task.id
                and run.event_id == task.event_id
                and run.organization_id == task.organization_id
                and run.user_id == task.user_id
            )
            if run_matches and run is not None and run.status in ACTIVE_RUN_STATUSES:
                continue

            event = await session.get(Event, task.event_id)
            if event is None:
                continue
            if run_matches and run is not None and run.status == "cancelled":
                target_status = "cancelled"
                reason = "The Task's Run was cancelled before Task state converged."
            elif (
                run_matches
                and run is not None
                and run.status == "completed"
                and task.status == "in_progress"
                and await task_has_pending_proposal(session, run_id=run.id)
            ):
                target_status = "waiting_approval"
                reason = "The Task is waiting for its protected action decision."
            elif (
                run_matches
                and run is not None
                and run.status == "completed"
                and run.success
            ):
                target_status = "done"
                reason = run.final_answer or "The Task's Run completed successfully."
            else:
                target_status = "failed"
                reason = (
                    "The Task lost its durable Run and was made retryable."
                    if run is None
                    else "The Task's Run ended before Task state converged."
                )

            changes: dict[str, Any] = {"status": target_status, "blocker": ""}
            if target_status in {"done", "failed"}:
                changes["result_summary"] = reason[:50_000]
            source_run_id = run.id if run_matches and run is not None else None
            await mutate_task(
                session,
                event=event,
                task=task,
                changes=changes,
                actor_id="worker:task-watchdog",
                source_run_id=source_run_id,
            )
            session.add(
                EventTrailEntry(
                    organization_id=task.organization_id,
                    user_id=task.user_id,
                    event_id=task.event_id,
                    actor_id="worker:task-watchdog",
                    kind="task_watchdog_reconciled",
                    summary=f"Reconciled stalled task {task.title}",
                    run_id=source_run_id,
                    task_id=task.id,
                    payload={
                        "status": target_status,
                        "reason": reason,
                    },
                )
            )
            await session.commit()
            reconciled += 1
    return reconciled


__all__ = [
    "reconcile_orphaned_tasks",
    "reconcile_stale_runs",
]
