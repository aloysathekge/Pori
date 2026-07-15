"""Agent tools for reversible Event-owned Task working state."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from pori import RunContext

from ..database import async_session
from ..event_presenters import task_payload
from ..models import Event, EventTrailEntry, Task
from ..task_state import (
    TaskBudgetPolicy,
    TaskExecutionMode,
    TaskPriority,
    TaskStateError,
    TaskStatus,
    mutate_task,
    resolve_task_origin,
    task_snapshot,
)


class TaskCreateParams(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=1000)
    instructions: str = Field(default="", max_length=50_000)
    definition_of_done: str = Field(default="", max_length=10_000)
    priority: TaskPriority = "normal"
    due_at: datetime | None = None
    execution_mode: TaskExecutionMode = "manual"
    assigned_agent_id: str | None = Field(default=None, max_length=200)
    origin_conversation_id: str | None = None
    budget_policy: TaskBudgetPolicy = Field(default_factory=TaskBudgetPolicy)
    order: int | None = None


class TaskUpdateParams(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    task_id: str = Field(min_length=1)
    title: str | None = Field(default=None, min_length=1, max_length=1000)
    status: TaskStatus | None = None
    instructions: str | None = Field(default=None, max_length=50_000)
    definition_of_done: str | None = Field(default=None, max_length=10_000)
    priority: TaskPriority | None = None
    due_at: datetime | None = None
    execution_mode: TaskExecutionMode | None = None
    assigned_agent_id: str | None = Field(default=None, max_length=200)
    origin_conversation_id: str | None = None
    result_summary: str | None = Field(default=None, max_length=50_000)
    blocker: str | None = Field(default=None, max_length=10_000)
    budget_policy: TaskBudgetPolicy | None = None
    order: int | None = None


class TaskMutationHandler:
    """Persist agent Task mutations on the database engine's owning loop."""

    def __init__(
        self,
        *,
        run_context: RunContext,
        session_factory: Any = async_session,
        owner_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._run_context = run_context
        self._session_factory = session_factory
        self._owner_loop = owner_loop

    async def _on_owner_loop(self, coroutine):
        current = asyncio.get_running_loop()
        if self._owner_loop is None or self._owner_loop is current:
            return await coroutine
        future = asyncio.run_coroutine_threadsafe(coroutine, self._owner_loop)
        return await asyncio.wrap_future(future)

    async def _load_event(self, session: AsyncSession) -> Event:
        event_id = self._run_context.event_id
        if not event_id:
            raise ValueError("Event identity is required")
        event = await session.get(Event, event_id)
        if (
            event is None
            or event.organization_id != self._run_context.organization_id
            or event.user_id != self._run_context.user_id
        ):
            raise ValueError("Event is unavailable")
        if event.lifecycle == "archived":
            raise ValueError("Event is archived")
        return event

    async def _create(self, params: TaskCreateParams) -> dict[str, Any]:
        async with self._session_factory() as session:
            event = await self._load_event(session)
            order = params.order
            if order is None:
                order = (
                    await session.execute(
                        select(func.coalesce(func.max(col(Task.order)), -1)).where(
                            Task.event_id == event.id,
                            Task.organization_id == event.organization_id,
                            Task.user_id == event.user_id,
                        )
                    )
                ).scalar_one() + 1
            task = Task(
                organization_id=event.organization_id,
                user_id=event.user_id,
                event_id=event.id,
                origin_conversation_id=await self._resolve_origin(
                    session,
                    event=event,
                    explicit=params.origin_conversation_id,
                ),
                title=params.title,
                instructions=params.instructions,
                definition_of_done=params.definition_of_done,
                priority=params.priority,
                due_at=params.due_at,
                execution_mode=params.execution_mode,
                assigned_agent_id=params.assigned_agent_id,
                budget_policy=params.budget_policy.model_dump(exclude_none=True),
                order=order,
                created_by=self._run_context.agent_id,
            )
            event.updated_at = datetime.now(timezone.utc)
            session.add(event)
            session.add(task)
            session.add(
                EventTrailEntry(
                    organization_id=event.organization_id,
                    user_id=event.user_id,
                    event_id=event.id,
                    actor_id=self._run_context.agent_id,
                    kind="task_changed",
                    summary=f"Created task {task.title}",
                    run_id=self._run_context.run_id,
                    task_id=task.id,
                    evidence_refs=(
                        [{"conversation_id": task.origin_conversation_id}]
                        if task.origin_conversation_id
                        else []
                    ),
                    payload={"action": "created", "after": task_snapshot(task)},
                )
            )
            await session.commit()
            await session.refresh(task)
            return task_payload(task)

    async def create(self, params: TaskCreateParams) -> dict[str, Any]:
        return await self._on_owner_loop(self._create(params))

    async def _resolve_origin(
        self,
        session: AsyncSession,
        *,
        event: Event,
        explicit: str | None,
    ) -> str | None:
        if explicit is not None:
            return await resolve_task_origin(
                session,
                event=event,
                preferred_conversation_id=explicit,
            )
        try:
            return await resolve_task_origin(
                session,
                event=event,
                preferred_conversation_id=self._run_context.session_id,
            )
        except TaskStateError:
            return await resolve_task_origin(session, event=event)

    async def _update(self, params: TaskUpdateParams) -> dict[str, Any]:
        submitted = params.model_dump(exclude={"task_id"}, exclude_unset=True)
        nullable_fields = {"due_at", "assigned_agent_id", "origin_conversation_id"}
        changes = {
            key: value
            for key, value in submitted.items()
            if value is not None or key in nullable_fields
        }
        if not changes:
            raise ValueError("At least one Task change is required")
        async with self._session_factory() as session:
            event = await self._load_event(session)
            task = await session.get(Task, params.task_id)
            if (
                task is None
                or task.event_id != event.id
                or task.organization_id != event.organization_id
                or task.user_id != event.user_id
            ):
                raise ValueError("Task is unavailable")
            if "budget_policy" in changes:
                changes["budget_policy"] = dict(changes["budget_policy"])
            await mutate_task(
                session,
                event=event,
                task=task,
                changes=changes,
                actor_id=self._run_context.agent_id,
                source_run_id=self._run_context.run_id,
            )
            await session.commit()
            await session.refresh(task)
            return task_payload(task)

    async def update(self, params: TaskUpdateParams) -> dict[str, Any]:
        return await self._on_owner_loop(self._update(params))


async def task_create_tool(params: TaskCreateParams, context: dict) -> dict[str, Any]:
    handler = context.get("task_mutator")
    if not isinstance(handler, TaskMutationHandler):
        raise ValueError("Task mutation is unavailable for this run")
    return await handler.create(params)


async def task_update_tool(params: TaskUpdateParams, context: dict) -> dict[str, Any]:
    handler = context.get("task_mutator")
    if not isinstance(handler, TaskMutationHandler):
        raise ValueError("Task mutation is unavailable for this run")
    return await handler.update(params)


def register_task_tools(registry) -> None:
    if "task_create" not in registry.tools:
        registry.register_tool(
            name="task_create",
            param_model=TaskCreateParams,
            function=task_create_tool,
            description=(
                "Create a reversible task in the current Event. This updates "
                "internal working state directly and records it in the Trail."
            ),
        )
    if "task_update" not in registry.tools:
        registry.register_tool(
            name="task_update",
            param_model=TaskUpdateParams,
            function=task_update_tool,
            description=(
                "Update a task in the current Event through its legal state "
                "machine and record the change in the Trail."
            ),
        )


__all__ = [
    "TaskCreateParams",
    "TaskMutationHandler",
    "TaskUpdateParams",
    "register_task_tools",
    "task_create_tool",
    "task_update_tool",
]
