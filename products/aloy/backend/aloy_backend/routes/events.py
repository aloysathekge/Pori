"""Event-owned Proposal decision endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import col, select

from ..database import get_session
from ..event_presenters import (
    event_payload,
    file_payload,
    proposal_payload,
    task_payload,
    trail_payload,
)
from ..events import ensure_event_conversation, ensure_life_event
from ..models import (
    ActionProposal,
    Conversation,
    Event,
    EventTrailEntry,
    StoredFile,
    Task,
)
from ..proposal_executor import (
    ProposalDecisionError,
    decide_proposal,
    execute_proposal,
)
from ..rate_limit import rate_limited_permission
from ..task_execution import (
    TaskExecutionError,
    queue_task_run,
    stop_task_run,
)
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
from ..tenancy import OrganizationContext, Permission

router = APIRouter(prefix="/events", tags=["events"])


class EventCreateBody(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=300)
    summary: str = Field(default="", max_length=2000)
    phase: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=50_000)
    origin_conversation_id: str | None = None


class TaskCreateBody(BaseModel):
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


class TaskUpdateBody(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

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


class TaskResumeBody(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    response: str | None = Field(default=None, max_length=50_000)


async def _load_event(
    session: AsyncSession,
    context: OrganizationContext,
    event_id: str,
) -> Event:
    event = await session.get(Event, event_id)
    if (
        event is None
        or event.organization_id != context.organization_id
        or event.user_id != context.user_id
    ):
        raise HTTPException(status_code=404, detail="Event not found")
    return event


async def _load_task(
    session: AsyncSession,
    context: OrganizationContext,
    event_id: str,
    task_id: str,
) -> Task:
    task = await session.get(Task, task_id)
    if (
        task is None
        or task.event_id != event_id
        or task.organization_id != context.organization_id
        or task.user_id != context.user_id
    ):
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post("", status_code=201)
async def create_event(
    body: EventCreateBody,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.AGENT_WRITE)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Create a Project Event directly under the user's authority."""
    origin: Conversation | None = None
    if body.origin_conversation_id:
        origin = await session.get(Conversation, body.origin_conversation_id)
        if (
            origin is None
            or origin.organization_id != context.organization_id
            or origin.user_id != context.user_id
        ):
            raise HTTPException(status_code=404, detail="Origin conversation not found")
        origin_event = await session.get(Event, origin.event_id)
        if origin_event is None or not origin_event.is_life:
            raise HTTPException(
                status_code=409,
                detail="Only a Life conversation can seed a new Event",
            )
    event = Event(
        organization_id=context.organization_id,
        user_id=context.user_id,
        type="project",
        title=body.title.strip(),
        lifecycle="active",
        phase=body.phase.strip(),
        summary=body.summary.strip(),
        metadata_={
            "notes": body.notes,
            **({"origin_conversation_id": origin.id} if origin is not None else {}),
        },
    )
    session.add(event)
    await session.flush()
    await ensure_event_conversation(session, event=event)
    session.add(
        EventTrailEntry(
            organization_id=context.organization_id,
            user_id=context.user_id,
            event_id=event.id,
            actor_id=context.user_id,
            kind="event_created",
            summary=f"Created Project Event {event.title}",
            evidence_refs=(
                [{"conversation_id": origin.id}] if origin is not None else []
            ),
            payload={
                "type": "project",
                "lifecycle": "active",
                **({"origin_conversation_id": origin.id} if origin is not None else {}),
            },
        )
    )
    await session.commit()
    await session.refresh(event)
    return event_payload(event)


@router.get("")
async def list_events(
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_READ)
    ),
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    life = await ensure_life_event(
        session,
        organization_id=context.organization_id,
        user_id=context.user_id,
    )
    await session.commit()
    events = (
        (
            await session.execute(
                select(Event)
                .where(
                    Event.organization_id == context.organization_id,
                    Event.user_id == context.user_id,
                    Event.lifecycle != "archived",
                )
                .order_by(col(Event.is_life).desc(), col(Event.updated_at).desc())
            )
        )
        .scalars()
        .all()
    )
    return [event_payload(event) for event in events]


@router.get("/{event_id}")
async def get_event_surface(
    event_id: str,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_READ)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Recompute the trusted Event Surface from durable rows on every read."""
    event = await _load_event(session, context, event_id)
    if not event.is_life:
        await ensure_event_conversation(session, event=event)
    await session.commit()
    tasks = (
        (
            await session.execute(
                select(Task)
                .where(
                    Task.event_id == event.id,
                    Task.organization_id == context.organization_id,
                    Task.user_id == context.user_id,
                )
                .order_by(col(Task.order), col(Task.created_at))
            )
        )
        .scalars()
        .all()
    )
    entries = (
        (
            await session.execute(
                select(EventTrailEntry)
                .where(
                    EventTrailEntry.event_id == event.id,
                    EventTrailEntry.organization_id == context.organization_id,
                    EventTrailEntry.user_id == context.user_id,
                )
                .order_by(col(EventTrailEntry.created_at).desc())
            )
        )
        .scalars()
        .all()
    )
    files = (
        (
            await session.execute(
                select(StoredFile)
                .where(
                    StoredFile.event_id == event.id,
                    StoredFile.organization_id == context.organization_id,
                    StoredFile.user_id == context.user_id,
                )
                .order_by(col(StoredFile.created_at).desc())
            )
        )
        .scalars()
        .all()
    )
    proposals = (
        (
            await session.execute(
                select(ActionProposal)
                .where(
                    ActionProposal.event_id == event.id,
                    ActionProposal.organization_id == context.organization_id,
                    ActionProposal.user_id == context.user_id,
                    ActionProposal.status == "pending",
                )
                .order_by(col(ActionProposal.created_at).desc())
            )
        )
        .scalars()
        .all()
    )
    return {
        "event": event_payload(event),
        "surface": {
            "type": event.type,
            "sections": [
                {
                    "kind": "status",
                    "summary": event.summary,
                    "phase": event.phase,
                },
                {"kind": "tasks", "tasks": [task_payload(task) for task in tasks]},
                {
                    "kind": "activity",
                    "entries": [trail_payload(entry) for entry in entries],
                },
                {
                    "kind": "notes",
                    "notes": str((event.metadata_ or {}).get("notes") or ""),
                },
                {
                    "kind": "files",
                    "files": [file_payload(file) for file in files],
                },
            ],
            "proposals": [proposal_payload(proposal) for proposal in proposals],
        },
    }


@router.post("/{event_id}/tasks", status_code=201)
async def create_task(
    event_id: str,
    body: TaskCreateBody,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_CREATE)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    event = await _load_event(session, context, event_id)
    if event.lifecycle == "archived":
        raise HTTPException(status_code=409, detail="Event is archived")
    order = body.order
    if order is None:
        order = (
            await session.execute(
                select(func.coalesce(func.max(col(Task.order)), -1)).where(
                    Task.event_id == event.id,
                    Task.organization_id == context.organization_id,
                    Task.user_id == context.user_id,
                )
            )
        ).scalar_one() + 1
    try:
        origin_conversation_id = await resolve_task_origin(
            session,
            event=event,
            preferred_conversation_id=body.origin_conversation_id,
        )
    except TaskStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    task = Task(
        organization_id=context.organization_id,
        user_id=context.user_id,
        event_id=event.id,
        origin_conversation_id=origin_conversation_id,
        title=body.title.strip(),
        instructions=body.instructions,
        definition_of_done=body.definition_of_done,
        priority=body.priority,
        due_at=body.due_at,
        execution_mode=body.execution_mode,
        assigned_agent_id=body.assigned_agent_id,
        budget_policy=body.budget_policy.model_dump(exclude_none=True),
        order=order,
        created_by="user",
    )
    event.updated_at = datetime.now(timezone.utc)
    session.add(event)
    session.add(task)
    session.add(
        EventTrailEntry(
            organization_id=context.organization_id,
            user_id=context.user_id,
            event_id=event.id,
            actor_id=context.user_id,
            kind="task_changed",
            summary=f"Created task {task.title}",
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


@router.patch("/{event_id}/tasks/{task_id}")
async def update_task(
    event_id: str,
    task_id: str,
    body: TaskUpdateBody,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_CREATE)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    event = await _load_event(session, context, event_id)
    if event.lifecycle == "archived":
        raise HTTPException(status_code=409, detail="Event is archived")
    task = await _load_task(session, context, event_id, task_id)
    submitted = body.model_dump(exclude_unset=True)
    nullable_fields = {"due_at", "assigned_agent_id", "origin_conversation_id"}
    changes = {
        key: value
        for key, value in submitted.items()
        if value is not None or key in nullable_fields
    }
    if not changes:
        raise HTTPException(status_code=422, detail="No Task changes provided")
    if "budget_policy" in changes:
        changes["budget_policy"] = dict(changes["budget_policy"])
    try:
        await mutate_task(
            session,
            event=event,
            task=task,
            changes=changes,
            actor_id=context.user_id,
        )
    except TaskStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    await session.commit()
    await session.refresh(task)
    return task_payload(task)


def _task_execution_payload(result) -> dict[str, Any]:
    return {
        "task": task_payload(result.task),
        "run": {
            "id": result.run.id,
            "status": result.run.status,
            "conversation_id": result.run.conversation_id,
            "attempt_count": result.run.attempt_count,
        },
        "idempotent": result.idempotent,
    }


@router.post("/{event_id}/tasks/{task_id}/work", status_code=202)
async def work_on_task(
    event_id: str,
    task_id: str,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_CREATE)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Queue explicit Task work on the durable worker."""
    event = await _load_event(session, context, event_id)
    task = await _load_task(session, context, event_id, task_id)
    try:
        result = await queue_task_run(
            session,
            event=event,
            task=task,
            context=context,
            control="work",
        )
    except (TaskExecutionError, TaskStateError) as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _task_execution_payload(result)


@router.post("/{event_id}/tasks/{task_id}/retry", status_code=202)
async def retry_task(
    event_id: str,
    task_id: str,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_CREATE)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    event = await _load_event(session, context, event_id)
    task = await _load_task(session, context, event_id, task_id)
    try:
        result = await queue_task_run(
            session,
            event=event,
            task=task,
            context=context,
            control="retry",
        )
    except (TaskExecutionError, TaskStateError) as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _task_execution_payload(result)


@router.post("/{event_id}/tasks/{task_id}/resume", status_code=202)
async def resume_task(
    event_id: str,
    task_id: str,
    body: TaskResumeBody,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_CREATE)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    event = await _load_event(session, context, event_id)
    task = await _load_task(session, context, event_id, task_id)
    try:
        result = await queue_task_run(
            session,
            event=event,
            task=task,
            context=context,
            control="resume",
            response=body.response,
        )
    except (TaskExecutionError, TaskStateError) as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _task_execution_payload(result)


@router.post("/{event_id}/tasks/{task_id}/stop")
async def stop_task(
    event_id: str,
    task_id: str,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_CANCEL)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    event = await _load_event(session, context, event_id)
    task = await _load_task(session, context, event_id, task_id)
    try:
        result = await stop_task_run(
            session,
            event=event,
            task=task,
            actor_id=context.user_id,
        )
    except (TaskExecutionError, TaskStateError) as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if result is None:
        return {"task": task_payload(task), "run": None, "idempotent": True}
    return _task_execution_payload(result)


@router.delete("/{event_id}/tasks/{task_id}", status_code=204)
async def delete_task(
    event_id: str,
    task_id: str,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_CREATE)
    ),
    session: AsyncSession = Depends(get_session),
) -> None:
    event = await _load_event(session, context, event_id)
    if event.lifecycle == "archived":
        raise HTTPException(status_code=409, detail="Event is archived")
    task = await _load_task(session, context, event_id, task_id)
    if task.status in {"queued", "in_progress", "blocked", "waiting_approval"}:
        raise HTTPException(
            status_code=409,
            detail="Stop active Task work before deleting it",
        )
    session.add(
        EventTrailEntry(
            organization_id=context.organization_id,
            user_id=context.user_id,
            event_id=event_id,
            actor_id=context.user_id,
            kind="task_changed",
            summary=f"Deleted task {task.title}",
            task_id=task.id,
            payload={"action": "deleted", "title": task.title},
        )
    )
    event.updated_at = datetime.now(timezone.utc)
    session.add(event)
    await session.delete(task)
    await session.commit()


class ProposalDecisionBody(BaseModel):
    decision: Literal["approve", "reject", "edit"]
    message: str | None = None
    edited_action: dict[str, Any] | None = None


@router.post("/{event_id}/proposals/{proposal_id}/decision")
async def submit_proposal_decision(
    event_id: str,
    proposal_id: str,
    body: ProposalDecisionBody,
    background_tasks: BackgroundTasks,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_CREATE)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Persist a decision; approved Proposals execute on the same dumb rail."""
    try:
        proposal = await decide_proposal(
            session,
            event_id=event_id,
            proposal_id=proposal_id,
            organization_id=context.organization_id,
            user_id=context.user_id,
            actor_id=context.user_id,
            decision=body.decision,
            edited_action=body.edited_action,
            message=body.message,
        )
    except ProposalDecisionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    if proposal.status == "approved":
        # Use a fresh session bound to the request's engine. This works for the
        # production engine and for dependency-overridden test engines.
        session_factory = async_sessionmaker(
            session.bind,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        background_tasks.add_task(
            execute_proposal,
            proposal.id,
            session_factory=session_factory,
        )
    return {
        "proposal_id": proposal.id,
        "status": proposal.status,
        "decision": body.decision,
    }


__all__ = ["ProposalDecisionBody", "router", "submit_proposal_decision"]
