"""Run endpoints: submit a run (202 — executed asynchronously by the durable
worker), list/get runs, read the run event log, create child runs, and cancel.
Operates on ``Run`` / ``RunEventLog`` rows; tenancy-gated via
``require_permission``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import col, select

from ..database import get_session
from ..events import ensure_life_event
from ..models import Run, RunEventLog, RunTimelineEvent
from ..run_budgets import narrow_budget_to_parent, resolve_run_budget
from ..schemas import (
    ChildRunCreate,
    RunEventLogResponse,
    RunRequest,
    RunResponse,
    RunTimelinePage,
)
from ..tenancy import OrganizationContext, Permission, require_permission

logger = logging.getLogger("aloy_backend")

router = APIRouter(prefix="/runs", tags=["runs"])


def _run_response(run: Run) -> RunResponse:
    return RunResponse(
        id=run.id,
        organization_id=run.organization_id,
        event_id=run.event_id,
        task_id=run.task_id,
        agent_id=run.agent_id,
        session_id=run.session_id,
        status=run.status,
        max_steps=run.max_steps,
        max_tool_calls=run.max_tool_calls,
        max_tokens=run.max_tokens,
        max_cost_usd=run.max_cost_usd,
        success=run.success,
        steps_taken=run.steps_taken,
        final_answer=run.final_answer,
        reasoning=run.reasoning,
        metrics=run.metrics,
        prompt_fingerprint=run.prompt_fingerprint,
        tool_surface_fingerprint=run.tool_surface_fingerprint,
        execution_receipts=run.execution_receipts,
        model_assignment=run.model_assignment,
        selected_skills=run.selected_skills,
        artifacts=run.artifacts,
        plan=run.plan,
        parent_run_id=run.parent_run_id,
        root_run_id=run.root_run_id,
        idempotency_key=run.idempotency_key,
        child_depth=run.child_depth,
        attempt_count=run.attempt_count,
        max_attempts=run.max_attempts,
        timeout_seconds=run.timeout_seconds,
        lease_owner=run.lease_owner,
        lease_expires_at=run.lease_expires_at,
        cancel_requested=run.cancel_requested,
        isolation_profile=run.isolation_profile,
        created_at=run.created_at,
    )


@router.post("", response_model=RunResponse, status_code=202)
async def create_run(
    req: RunRequest,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
) -> RunResponse:
    """Create a run that executes in the background. Poll GET /runs/{id} for status."""
    if not req.task.strip():
        raise HTTPException(status_code=400, detail="task is required")

    active_count = (
        await session.execute(
            select(func.count())
            .select_from(Run)
            .where(
                Run.organization_id == context.organization_id,
                col(Run.status).in_(["pending", "running"]),
            )
        )
    ).scalar_one()
    if active_count >= context.policy.max_concurrent_runs:
        raise HTTPException(status_code=429, detail="Organization run limit reached")

    budget = resolve_run_budget(
        context.policy,
        req.model_dump(exclude={"task"}),
        default_max_steps=req.max_steps,
    )
    life = await ensure_life_event(
        session,
        organization_id=context.organization_id,
        user_id=context.user_id,
    )
    run = Run(
        user_id=context.user_id,
        organization_id=context.organization_id,
        event_id=life.id,
        agent_id="default_agent",
        session_id="pending",
        task=req.task,
        max_steps=budget.max_steps,
        max_tool_calls=budget.max_tool_calls,
        max_tokens=budget.max_tokens,
        max_cost_usd=budget.max_cost_usd,
        status="pending",
        max_attempts=context.policy.max_attempts,
        timeout_seconds=budget.timeout_seconds,
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)
    run.session_id = run.id
    run.root_run_id = run.id
    session.add(run)
    await session.commit()

    logger.info("Run %s queued", run.id)

    return _run_response(run)


@router.get("", response_model=list[RunResponse])
async def list_runs(
    context: OrganizationContext = Depends(require_permission(Permission.RUN_READ)),
    session: AsyncSession = Depends(get_session),
) -> Sequence[Run]:
    result = await session.execute(
        select(Run)
        .where(Run.organization_id == context.organization_id)
        .order_by(col(Run.created_at).desc())
    )
    return result.scalars().all()


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_READ)),
    session: AsyncSession = Depends(get_session),
) -> Run:
    run = await session.get(Run, run_id)
    if not run or run.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/{run_id}/events", response_model=RunEventLogResponse)
async def get_run_events(
    run_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_READ)),
    session: AsyncSession = Depends(get_session),
) -> RunEventLog:
    """The coalesced event log for a run — powers the read-only replay view."""
    log = await session.get(RunEventLog, run_id)
    if not log or log.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Run event log not found")
    return log


def _timeline_payload(entry: RunTimelineEvent) -> dict[str, Any]:
    return {
        "id": entry.id,
        "run_id": entry.run_id,
        "sequence": entry.sequence,
        "kind": entry.kind,
        "public_payload": entry.public_payload,
        "created_at": entry.created_at,
    }


@router.get("/{run_id}/timeline", response_model=RunTimelinePage)
async def get_run_timeline(
    run_id: str,
    after: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=500),
    context: OrganizationContext = Depends(require_permission(Permission.RUN_READ)),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Return durable, user-safe Work Story milestones after a sequence cursor."""
    run = await session.get(Run, run_id)
    if not run or run.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Run not found")
    rows = list(
        (
            await session.execute(
                select(RunTimelineEvent)
                .where(
                    RunTimelineEvent.run_id == run_id,
                    RunTimelineEvent.organization_id == context.organization_id,
                    RunTimelineEvent.sequence > after,
                )
                .order_by(col(RunTimelineEvent.sequence))
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return {
        "entries": [_timeline_payload(row) for row in rows],
        "next_cursor": rows[-1].sequence if rows else after,
    }


def _timeline_sse(
    event: str, data: dict[str, Any], *, event_id: int | None = None
) -> str:
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    encoded = json.dumps(jsonable_encoder(data), separators=(",", ":"))
    return f"{prefix}event: {event}\ndata: {encoded}\n\n"


@router.get("/{run_id}/live")
async def stream_run_timeline(
    run_id: str,
    request: Request,
    cursor: int = Query(0, ge=0),
    context: OrganizationContext = Depends(require_permission(Permission.RUN_READ)),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Replay durable Work Story milestones, then follow the Run until terminal."""
    run = await session.get(Run, run_id)
    if not run or run.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Run not found")
    bind = session.bind
    await session.rollback()
    session_factory = async_sessionmaker(
        bind, class_=AsyncSession, expire_on_commit=False
    )
    organization_id = context.organization_id

    async def generate():
        current = cursor
        last_heartbeat = time.monotonic()
        yield _timeline_sse("ready", {"run_id": run_id, "cursor": current})
        while not await request.is_disconnected():
            async with session_factory() as live_session:
                rows = list(
                    (
                        await live_session.execute(
                            select(RunTimelineEvent)
                            .where(
                                RunTimelineEvent.run_id == run_id,
                                RunTimelineEvent.organization_id == organization_id,
                                RunTimelineEvent.sequence > current,
                            )
                            .order_by(col(RunTimelineEvent.sequence))
                            .limit(200)
                        )
                    )
                    .scalars()
                    .all()
                )
                current_run = await live_session.get(Run, run_id)
            for row in rows:
                current = row.sequence
                yield _timeline_sse(
                    "timeline_event",
                    _timeline_payload(row),
                    event_id=row.sequence,
                )
            if (
                current_run
                and current_run.status
                in {
                    "completed",
                    "failed",
                    "cancelled",
                }
                and not rows
            ):
                yield _timeline_sse("done", {"cursor": current})
                break
            if time.monotonic() - last_heartbeat >= 15:
                yield _timeline_sse("heartbeat", {"cursor": current})
                last_heartbeat = time.monotonic()
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{run_id}/children", response_model=RunResponse, status_code=202)
async def create_child_run(
    run_id: str,
    body: ChildRunCreate,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
) -> RunResponse:
    parent = await session.get(Run, run_id)
    if not parent or parent.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Parent run not found")
    if parent.child_depth >= context.policy.max_child_depth:
        raise HTTPException(status_code=409, detail="Child run depth limit reached")
    child_count = (
        await session.execute(
            select(func.count())
            .select_from(Run)
            .where(
                Run.organization_id == context.organization_id,
                Run.parent_run_id == parent.id,
            )
        )
    ).scalar_one()
    if child_count >= context.policy.max_child_runs_per_run:
        raise HTTPException(status_code=429, detail="Child run limit reached")
    if body.idempotency_key:
        existing = await session.execute(
            select(Run).where(
                Run.organization_id == context.organization_id,
                Run.parent_run_id == parent.id,
                Run.idempotency_key == body.idempotency_key,
            )
        )
        row = existing.scalars().first()
        if row is not None:
            return _run_response(row)
    budget = narrow_budget_to_parent(
        resolve_run_budget(
            context.policy,
            body.model_dump(exclude={"task", "agent_id", "idempotency_key"}),
            default_max_steps=body.max_steps,
        ),
        parent,
    )
    child = Run(
        user_id=context.user_id,
        organization_id=context.organization_id,
        event_id=parent.event_id,
        task_id=parent.task_id,
        agent_id=body.agent_id,
        session_id="pending",
        task=body.task,
        max_steps=budget.max_steps,
        max_tool_calls=budget.max_tool_calls,
        max_tokens=budget.max_tokens,
        max_cost_usd=budget.max_cost_usd,
        status="pending",
        max_attempts=context.policy.max_attempts,
        timeout_seconds=budget.timeout_seconds,
        parent_run_id=parent.id,
        root_run_id=parent.root_run_id or parent.id,
        idempotency_key=body.idempotency_key,
        child_depth=parent.child_depth + 1,
    )
    session.add(child)
    await session.commit()
    await session.refresh(child)
    child.session_id = child.id
    session.add(child)
    await session.commit()
    await session.refresh(child)
    return _run_response(child)


@router.post("/{run_id}/cancel", response_model=RunResponse)
async def cancel_run(
    run_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CANCEL)),
    session: AsyncSession = Depends(get_session),
) -> Run:
    run = await session.get(Run, run_id)
    if not run or run.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status in {"completed", "failed", "cancelled"}:
        return run
    run.cancel_requested = True
    if run.status == "pending":
        run.status = "cancelled"
    session.add(run)
    await session.commit()
    await session.refresh(run)
    return run
