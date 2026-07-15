"""Agent tools for reversible Event-owned Task working state."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from pori import RunContext

from ..database import async_session
from ..event_presenters import task_payload
from ..models import Event, EventTrailEntry, Task


class TaskCreateParams(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=1000)
    order: int | None = None


class TaskUpdateParams(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    task_id: str = Field(min_length=1)
    title: str | None = Field(default=None, min_length=1, max_length=1000)
    status: Literal["open", "done"] | None = None
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
                title=params.title,
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
                    payload={"action": "created", "status": "open", "order": order},
                )
            )
            await session.commit()
            await session.refresh(task)
            return task_payload(task)

    async def create(self, params: TaskCreateParams) -> dict[str, Any]:
        return await self._on_owner_loop(self._create(params))

    async def _update(self, params: TaskUpdateParams) -> dict[str, Any]:
        changes = params.model_dump(
            exclude={"task_id"}, exclude_unset=True, exclude_none=True
        )
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
            before = {"title": task.title, "status": task.status, "order": task.order}
            if "title" in changes:
                task.title = str(changes["title"])
            if "status" in changes:
                task.status = str(changes["status"])
            if "order" in changes:
                task.order = int(changes["order"])
            task.updated_at = datetime.now(timezone.utc)
            event.updated_at = task.updated_at
            after = {"title": task.title, "status": task.status, "order": task.order}
            session.add(task)
            session.add(event)
            session.add(
                EventTrailEntry(
                    organization_id=event.organization_id,
                    user_id=event.user_id,
                    event_id=event.id,
                    actor_id=self._run_context.agent_id,
                    kind="task_changed",
                    summary=f"Updated task {task.title}",
                    run_id=self._run_context.run_id,
                    task_id=task.id,
                    payload={"action": "updated", "before": before, "after": after},
                )
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
                "Update, complete, reopen, rename, or reorder a task in the "
                "current Event and record the change in the Trail."
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
