"""Executable Task invariants, transitions, provenance, and atomic claims."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import or_, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from .models import Conversation, Event, EventTrailEntry, Run, Task

TaskStatus = Literal[
    "open",
    "queued",
    "in_progress",
    "blocked",
    "waiting_approval",
    "done",
    "failed",
    "cancelled",
]
TaskPriority = Literal["low", "normal", "high", "urgent"]
TaskExecutionMode = Literal["manual"]


class TaskBudgetPolicy(BaseModel):
    """Optional per-Task ceilings; organization policy remains authoritative."""

    model_config = ConfigDict(extra="forbid")

    max_steps: int | None = Field(default=None, ge=1, le=10_000)
    timeout_seconds: int | None = Field(default=None, ge=1, le=86_400)
    max_tool_calls: int | None = Field(default=None, ge=1, le=10_000)
    max_cost_usd: float | None = Field(default=None, ge=0, le=10_000)


TASK_STATUSES: tuple[TaskStatus, ...] = (
    "open",
    "queued",
    "in_progress",
    "blocked",
    "waiting_approval",
    "done",
    "failed",
    "cancelled",
)
TERMINAL_TASK_STATUSES = frozenset({"done", "failed", "cancelled"})
ACTIVE_TASK_STATUSES = frozenset(TASK_STATUSES) - TERMINAL_TASK_STATUSES

LEGAL_TASK_TRANSITIONS: dict[str, frozenset[str]] = {
    "open": frozenset({"queued", "done", "cancelled"}),
    "queued": frozenset({"in_progress", "failed", "cancelled"}),
    "in_progress": frozenset(
        {"blocked", "waiting_approval", "done", "failed", "cancelled"}
    ),
    "blocked": frozenset({"queued", "in_progress", "failed", "cancelled"}),
    "waiting_approval": frozenset(
        {"queued", "in_progress", "done", "failed", "cancelled"}
    ),
    "done": frozenset({"open"}),
    "failed": frozenset({"open", "queued"}),
    "cancelled": frozenset({"open"}),
}

MUTABLE_TASK_FIELDS = frozenset(
    {
        "title",
        "status",
        "instructions",
        "definition_of_done",
        "priority",
        "due_at",
        "execution_mode",
        "assigned_agent_id",
        "origin_conversation_id",
        "result_summary",
        "blocker",
        "budget_policy",
        "order",
    }
)


class TaskStateError(ValueError):
    """A Task mutation violates the executable-state contract."""


def task_snapshot(task: Task) -> dict[str, Any]:
    due_at = task.due_at
    if due_at is not None and due_at.tzinfo is None:
        due_at = due_at.replace(tzinfo=timezone.utc)
    return {
        "title": task.title,
        "status": task.status,
        "instructions": task.instructions,
        "definition_of_done": task.definition_of_done,
        "priority": task.priority,
        "due_at": (
            due_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            if due_at
            else None
        ),
        "execution_mode": task.execution_mode,
        "assigned_agent_id": task.assigned_agent_id,
        "origin_conversation_id": task.origin_conversation_id,
        "current_run_id": task.current_run_id,
        "result_summary": task.result_summary,
        "blocker": task.blocker,
        "budget_policy": task.budget_policy,
        "order": task.order,
    }


def validate_task_transition(current: str, target: str) -> None:
    if target not in TASK_STATUSES:
        raise TaskStateError(f"Unknown Task status: {target}")
    if current == target:
        return
    if target not in LEGAL_TASK_TRANSITIONS.get(current, frozenset()):
        raise TaskStateError(f"Illegal Task transition: {current} -> {target}")


async def validate_task_origin(
    session: AsyncSession,
    *,
    event: Event,
    conversation_id: str | None,
) -> str | None:
    if conversation_id is None:
        return None
    conversation = await session.get(Conversation, conversation_id)
    if (
        conversation is None
        or conversation.event_id != event.id
        or conversation.organization_id != event.organization_id
        or conversation.user_id != event.user_id
    ):
        raise TaskStateError("Task origin conversation must belong to its Event")
    return conversation.id


async def resolve_task_origin(
    session: AsyncSession,
    *,
    event: Event,
    preferred_conversation_id: str | None = None,
) -> str | None:
    candidate = preferred_conversation_id or event.primary_conversation_id
    return await validate_task_origin(
        session,
        event=event,
        conversation_id=candidate,
    )


async def mutate_task(
    session: AsyncSession,
    *,
    event: Event,
    task: Task,
    changes: dict[str, Any],
    actor_id: str,
    source_run_id: str | None = None,
) -> EventTrailEntry:
    unknown = set(changes) - MUTABLE_TASK_FIELDS
    if unknown:
        raise TaskStateError(f"Unsupported Task fields: {', '.join(sorted(unknown))}")
    if not changes:
        raise TaskStateError("At least one Task change is required")

    before = task_snapshot(task)
    if "status" in changes:
        validate_task_transition(task.status, str(changes["status"]))
    if "origin_conversation_id" in changes:
        changes["origin_conversation_id"] = await validate_task_origin(
            session,
            event=event,
            conversation_id=changes["origin_conversation_id"],
        )

    for field, value in changes.items():
        setattr(task, field, value)

    if task.status == "blocked" and not task.blocker.strip():
        raise TaskStateError("A blocked Task requires a blocker")
    if task.status in TERMINAL_TASK_STATUSES or task.status == "open":
        task.current_run_id = None

    now = datetime.now(timezone.utc)
    task.updated_at = now
    event.updated_at = now
    after = task_snapshot(task)
    entry = EventTrailEntry(
        organization_id=event.organization_id,
        user_id=event.user_id,
        event_id=event.id,
        actor_id=actor_id,
        kind="task_changed",
        summary=f"Updated task {task.title}",
        run_id=source_run_id,
        task_id=task.id,
        payload={"action": "updated", "before": before, "after": after},
    )
    session.add_all([task, event, entry])
    return entry


async def claim_task(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    event_id: str,
    task_id: str,
    run_id: str,
    actor_id: str,
) -> Task | None:
    """Atomically claim one queued Task; concurrent callers yield one winner."""
    now = datetime.now(timezone.utc)
    matching_run = (
        select(Run.id)
        .where(
            col(Run.id) == run_id,
            col(Run.organization_id) == organization_id,
            col(Run.user_id) == user_id,
            col(Run.event_id) == event_id,
            col(Run.task_id) == task_id,
            col(Run.status).in_(["pending", "running"]),
        )
        .exists()
    )
    result = await session.execute(
        update(Task)
        .where(
            col(Task.id) == task_id,
            col(Task.organization_id) == organization_id,
            col(Task.user_id) == user_id,
            col(Task.event_id) == event_id,
            col(Task.status) == "queued",
            or_(
                col(Task.current_run_id).is_(None),
                col(Task.current_run_id) == run_id,
            ),
            matching_run,
        )
        .values(status="in_progress", current_run_id=run_id, updated_at=now)
        .returning(col(Task.id))
    )
    if result.scalar_one_or_none() is None:
        await session.rollback()
        run = await session.get(Run, run_id)
        if (
            run is None
            or run.organization_id != organization_id
            or run.user_id != user_id
            or run.event_id != event_id
            or run.task_id != task_id
            or run.status not in {"pending", "running"}
        ):
            raise TaskStateError("Claim Run must belong to the Task and Event")
        return None

    task = await session.get(Task, task_id, populate_existing=True)
    event = await session.get(Event, event_id)
    if task is None or event is None:
        await session.rollback()
        raise TaskStateError("Claimed Task or Event disappeared")
    event.updated_at = now
    session.add(event)
    session.add(
        EventTrailEntry(
            organization_id=organization_id,
            user_id=user_id,
            event_id=event_id,
            actor_id=actor_id,
            kind="task_changed",
            summary=f"Started task {task.title}",
            run_id=run_id,
            task_id=task_id,
            payload={
                "action": "claimed",
                "before": {"status": "queued"},
                "after": {"status": "in_progress", "current_run_id": run_id},
            },
        )
    )
    await session.commit()
    await session.refresh(task)
    return task


__all__ = [
    "ACTIVE_TASK_STATUSES",
    "LEGAL_TASK_TRANSITIONS",
    "TASK_STATUSES",
    "TERMINAL_TASK_STATUSES",
    "TaskExecutionMode",
    "TaskBudgetPolicy",
    "TaskPriority",
    "TaskStateError",
    "TaskStatus",
    "claim_task",
    "mutate_task",
    "resolve_task_origin",
    "task_snapshot",
    "validate_task_origin",
    "validate_task_transition",
]
