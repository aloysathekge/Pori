from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ..database import get_session
from ..models import Run, RunEventLog
from ..schemas import ChildRunCreate, RunEventLogResponse, RunRequest, RunResponse
from ..tenancy import OrganizationContext, Permission, require_permission

logger = logging.getLogger("aloy_backend")

router = APIRouter(prefix="/runs", tags=["runs"])


def _run_response(run: Run) -> RunResponse:
    return RunResponse(
        id=run.id,
        organization_id=run.organization_id,
        agent_id=run.agent_id,
        session_id=run.session_id,
        status=run.status,
        max_steps=run.max_steps,
        success=run.success,
        steps_taken=run.steps_taken,
        final_answer=run.final_answer,
        reasoning=run.reasoning,
        metrics=run.metrics,
        prompt_fingerprint=run.prompt_fingerprint,
        tool_surface_fingerprint=run.tool_surface_fingerprint,
        execution_receipts=run.execution_receipts,
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
                Run.status.in_(["pending", "running"]),
            )
        )
    ).scalar_one()
    if active_count >= context.policy.max_concurrent_runs:
        raise HTTPException(status_code=429, detail="Organization run limit reached")

    max_steps = min(req.max_steps, context.policy.max_steps_per_run)
    run = Run(
        user_id=context.user_id,
        organization_id=context.organization_id,
        agent_id="default_agent",
        session_id="pending",
        task=req.task,
        max_steps=max_steps,
        status="pending",
        max_attempts=context.policy.max_attempts,
        timeout_seconds=context.policy.run_timeout_seconds,
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
):
    result = await session.execute(
        select(Run)
        .where(Run.organization_id == context.organization_id)
        .order_by(Run.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_READ)),
    session: AsyncSession = Depends(get_session),
):
    run = await session.get(Run, run_id)
    if not run or run.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/{run_id}/events", response_model=RunEventLogResponse)
async def get_run_events(
    run_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_READ)),
    session: AsyncSession = Depends(get_session),
):
    """The coalesced event log for a run — powers the read-only replay view."""
    log = await session.get(RunEventLog, run_id)
    if not log or log.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Run event log not found")
    return log


@router.post("/{run_id}/children", response_model=RunResponse, status_code=202)
async def create_child_run(
    run_id: str,
    body: ChildRunCreate,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
):
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
    max_steps = min(body.max_steps, context.policy.max_steps_per_run)
    child = Run(
        user_id=context.user_id,
        organization_id=context.organization_id,
        agent_id=body.agent_id,
        session_id="pending",
        task=body.task,
        max_steps=max_steps,
        status="pending",
        max_attempts=context.policy.max_attempts,
        timeout_seconds=context.policy.run_timeout_seconds,
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
):
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
