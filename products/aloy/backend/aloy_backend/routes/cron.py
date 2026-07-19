"""Cron job CRUD: recurring tasks that enqueue Runs on the worker queue."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from ..cron import compute_next_run, validate_schedule, validate_timezone
from ..database import get_session
from ..events import ensure_event_conversation
from ..models import CronJob, Event, EventTrailEntry, Run
from ..schemas import (
    CronJobCreate,
    CronJobResponse,
    CronJobUpdate,
    CronRunResponse,
)
from ..tenancy import OrganizationContext, Permission, require_permission

logger = logging.getLogger("aloy_backend")

router = APIRouter(prefix="/cron", tags=["cron"])

MAX_CRON_JOBS_PER_ORG = 50


async def _get_owned_job(
    job_id: str, context: OrganizationContext, session: AsyncSession
) -> CronJob:
    job = await session.get(CronJob, job_id)
    if (
        not job
        or job.organization_id != context.organization_id
        or job.deleted_at is not None
    ):
        raise HTTPException(status_code=404, detail="Cron job not found")
    return job


async def _get_owned_event(
    event_id: str,
    context: OrganizationContext,
    session: AsyncSession,
    *,
    require_active: bool = True,
) -> Event:
    event = await session.get(Event, event_id)
    if (
        event is None
        or event.organization_id != context.organization_id
        or event.user_id != context.user_id
    ):
        raise HTTPException(status_code=404, detail="Event not found")
    if event.is_life:
        raise HTTPException(
            status_code=409,
            detail="Schedules must belong to a dedicated Event",
        )
    if require_active and event.lifecycle != "active":
        raise HTTPException(
            status_code=409,
            detail="Schedules can only be created for an active Event",
        )
    return event


@router.post("", response_model=CronJobResponse, status_code=201)
async def create_cron_job(
    req: CronJobCreate,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
) -> CronJob:
    try:
        validate_schedule(req.schedule, req.timezone)
        validate_timezone(req.timezone)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    event = await _get_owned_event(req.event_id, context, session)
    conversation = await ensure_event_conversation(session, event=event)

    count = (
        await session.execute(
            select(func.count())
            .select_from(CronJob)
            .where(
                CronJob.organization_id == context.organization_id,
                col(CronJob.deleted_at).is_(None),
            )
        )
    ).scalar_one()
    if count >= MAX_CRON_JOBS_PER_ORG:
        raise HTTPException(status_code=429, detail="Cron job limit reached")

    job = CronJob(
        organization_id=context.organization_id,
        user_id=context.user_id,
        event_id=event.id,
        name=req.name,
        task=req.task,
        schedule=req.schedule.strip(),
        timezone=req.timezone.strip(),
        authority=req.authority,
        notification_mode=req.notification_mode,
        max_steps=min(req.max_steps, context.policy.max_steps_per_run),
        conversation_id=conversation.id,
        next_run_at=compute_next_run(req.schedule, timezone_name=req.timezone),
    )
    event.updated_at = datetime.now(timezone.utc)
    session.add(event)
    session.add(job)
    session.add(
        EventTrailEntry(
            organization_id=context.organization_id,
            user_id=context.user_id,
            event_id=event.id,
            actor_id=context.user_id,
            kind="schedule_created",
            summary=f"Created schedule: {job.name}",
            payload={
                "schedule_id": job.id,
                "schedule_name": job.name,
                "schedule": job.schedule,
                "timezone": job.timezone,
                "authority": job.authority,
                "notification_mode": job.notification_mode,
                "next_run_at": (
                    job.next_run_at.isoformat() if job.next_run_at else None
                ),
            },
        )
    )
    await session.commit()
    await session.refresh(job)
    logger.info("Cron job %s created (%s)", job.id, job.schedule)
    return job


@router.get("", response_model=list[CronJobResponse])
async def list_cron_jobs(
    limit: int = 100,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_READ)),
    session: AsyncSession = Depends(get_session),
) -> Sequence[CronJob]:
    result = await session.execute(
        select(CronJob)
        .where(
            CronJob.organization_id == context.organization_id,
            col(CronJob.deleted_at).is_(None),
        )
        .order_by(col(CronJob.created_at).desc())
        .limit(max(1, min(limit, 200)))
    )
    return result.scalars().all()


@router.patch("/{job_id}", response_model=CronJobResponse)
async def update_cron_job(
    job_id: str,
    req: CronJobUpdate,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
) -> CronJob:
    job = await _get_owned_job(job_id, context, session)
    if job.event_id is None:
        raise HTTPException(
            status_code=409,
            detail="This legacy schedule has no Event and must be recreated",
        )
    event = await _get_owned_event(job.event_id, context, session, require_active=False)
    previous_enabled = job.enabled
    schedule_changed = req.schedule is not None or req.timezone is not None
    next_schedule = req.schedule.strip() if req.schedule is not None else job.schedule
    next_timezone = req.timezone.strip() if req.timezone is not None else job.timezone
    if schedule_changed:
        try:
            validate_schedule(next_schedule, next_timezone)
            validate_timezone(next_timezone)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        job.schedule = next_schedule
        job.timezone = next_timezone
        job.next_run_at = compute_next_run(job.schedule, timezone_name=job.timezone)
    if req.name is not None:
        job.name = req.name
    if req.task is not None:
        job.task = req.task
    if req.authority is not None:
        job.authority = req.authority
    if req.notification_mode is not None:
        job.notification_mode = req.notification_mode
    if req.max_steps is not None:
        job.max_steps = min(req.max_steps, context.policy.max_steps_per_run)
    if req.enabled is not None:
        if req.enabled and not previous_enabled and event.lifecycle != "active":
            raise HTTPException(
                status_code=409,
                detail="Activate the Event before resuming this Schedule",
            )
        job.enabled = req.enabled
        if req.enabled and not previous_enabled:
            # Resume from now instead of replaying an occurrence missed while
            # the Event Schedule was deliberately paused.
            job.next_run_at = compute_next_run(job.schedule, timezone_name=job.timezone)
    job.updated_at = datetime.now(timezone.utc)
    event.updated_at = job.updated_at
    session.add(event)
    session.add(job)
    if req.enabled is not None and req.enabled != previous_enabled:
        kind = "schedule_resumed" if req.enabled else "schedule_paused"
        summary = f"{'Resumed' if req.enabled else 'Paused'} schedule: {job.name}"
    else:
        kind = "schedule_updated"
        summary = f"Updated schedule: {job.name}"
    session.add(
        EventTrailEntry(
            organization_id=context.organization_id,
            user_id=context.user_id,
            event_id=event.id,
            actor_id=context.user_id,
            kind=kind,
            summary=summary,
            payload={
                "schedule_id": job.id,
                "schedule_name": job.name,
                "enabled": job.enabled,
                "schedule": job.schedule,
                "timezone": job.timezone,
                "authority": job.authority,
                "notification_mode": job.notification_mode,
                "next_run_at": (
                    job.next_run_at.isoformat() if job.next_run_at else None
                ),
            },
        )
    )
    await session.commit()
    await session.refresh(job)
    return job


@router.delete("/{job_id}", status_code=204)
async def delete_cron_job(
    job_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
) -> None:
    job = await _get_owned_job(job_id, context, session)
    if job.event_id:
        event = await session.get(Event, job.event_id)
        if (
            event is not None
            and event.organization_id == context.organization_id
            and event.user_id == context.user_id
        ):
            event.updated_at = datetime.now(timezone.utc)
            session.add(event)
            session.add(
                EventTrailEntry(
                    organization_id=context.organization_id,
                    user_id=context.user_id,
                    event_id=event.id,
                    actor_id=context.user_id,
                    kind="schedule_deleted",
                    summary=f"Deleted schedule: {job.name}",
                    payload={
                        "schedule_id": job.id,
                        "schedule_name": job.name,
                    },
                )
            )
    job.enabled = False
    job.deleted_at = datetime.now(timezone.utc)
    job.updated_at = job.deleted_at
    session.add(job)
    await session.commit()
    return None


@router.get("/{job_id}/runs", response_model=list[CronRunResponse])
async def list_cron_runs(
    job_id: str,
    limit: int = 8,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_READ)),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    await _get_owned_job(job_id, context, session)
    runs = (
        (
            await session.execute(
                select(Run)
                .where(
                    Run.organization_id == context.organization_id,
                    Run.user_id == context.user_id,
                    Run.cron_job_id == job_id,
                )
                .order_by(col(Run.created_at).desc())
                .limit(max(1, min(limit, 25)))
            )
        )
        .scalars()
        .all()
    )
    return [
        {
            "id": run.id,
            "status": run.status,
            "success": run.success,
            "final_answer": (
                run.final_answer[:4000] if run.final_answer is not None else None
            ),
            "steps_taken": run.steps_taken,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
        }
        for run in runs
    ]
