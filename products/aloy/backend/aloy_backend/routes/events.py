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
from ..events import ensure_life_event
from ..models import ActionProposal, Event, EventTrailEntry, StoredFile, Task
from ..proposal_executor import (
    ProposalDecisionError,
    decide_proposal,
    execute_proposal,
)
from ..rate_limit import rate_limited_permission
from ..tenancy import OrganizationContext, Permission

router = APIRouter(prefix="/events", tags=["events"])


class EventCreateBody(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=300)
    summary: str = Field(default="", max_length=2000)
    phase: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=50_000)


class TaskCreateBody(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=1000)
    order: int | None = None


class TaskUpdateBody(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str | None = Field(default=None, min_length=1, max_length=1000)
    status: Literal["open", "done"] | None = None
    order: int | None = None


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
    event = Event(
        organization_id=context.organization_id,
        user_id=context.user_id,
        type="project",
        title=body.title.strip(),
        lifecycle="active",
        phase=body.phase.strip(),
        summary=body.summary.strip(),
        metadata_={"notes": body.notes},
    )
    session.add(event)
    session.add(
        EventTrailEntry(
            organization_id=context.organization_id,
            user_id=context.user_id,
            event_id=event.id,
            actor_id=context.user_id,
            kind="event_created",
            summary=f"Created Project Event {event.title}",
            payload={"type": "project", "lifecycle": "active"},
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
    await ensure_life_event(
        session,
        organization_id=context.organization_id,
        user_id=context.user_id,
    )
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
    task = Task(
        organization_id=context.organization_id,
        user_id=context.user_id,
        event_id=event.id,
        title=body.title.strip(),
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
            payload={"action": "created", "status": "open", "order": task.order},
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
    changes = body.model_dump(exclude_unset=True, exclude_none=True)
    if not changes:
        raise HTTPException(status_code=422, detail="No Task changes provided")
    before = {"title": task.title, "status": task.status, "order": task.order}
    if "title" in changes:
        task.title = str(changes["title"]).strip()
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
            organization_id=context.organization_id,
            user_id=context.user_id,
            event_id=event_id,
            actor_id=context.user_id,
            kind="task_changed",
            summary=f"Updated task {task.title}",
            task_id=task.id,
            payload={"action": "updated", "before": before, "after": after},
        )
    )
    await session.commit()
    await session.refresh(task)
    return task_payload(task)


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
