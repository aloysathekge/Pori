"""Cron job CRUD: recurring tasks that enqueue Runs on the worker queue."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ..cron import compute_next_run, validate_schedule
from ..database import get_session
from ..models import Conversation, CronJob
from ..schemas import CronJobCreate, CronJobResponse, CronJobUpdate
from ..tenancy import OrganizationContext, Permission, require_permission

logger = logging.getLogger("aloy_backend")

router = APIRouter(prefix="/cron", tags=["cron"])

MAX_CRON_JOBS_PER_ORG = 50


async def _get_owned_job(
    job_id: str, context: OrganizationContext, session: AsyncSession
) -> CronJob:
    job = await session.get(CronJob, job_id)
    if not job or job.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Cron job not found")
    return job


async def _check_conversation(
    conversation_id: str | None,
    context: OrganizationContext,
    session: AsyncSession,
) -> None:
    if conversation_id is None:
        return
    conversation = await session.get(Conversation, conversation_id)
    if not conversation or conversation.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Conversation not found")


@router.post("", response_model=CronJobResponse, status_code=201)
async def create_cron_job(
    req: CronJobCreate,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
):
    try:
        validate_schedule(req.schedule)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    await _check_conversation(req.conversation_id, context, session)

    count = (
        await session.execute(
            select(func.count())
            .select_from(CronJob)
            .where(CronJob.organization_id == context.organization_id)
        )
    ).scalar_one()
    if count >= MAX_CRON_JOBS_PER_ORG:
        raise HTTPException(status_code=429, detail="Cron job limit reached")

    job = CronJob(
        organization_id=context.organization_id,
        user_id=context.user_id,
        name=req.name,
        task=req.task,
        schedule=req.schedule.strip(),
        max_steps=min(req.max_steps, context.policy.max_steps_per_run),
        conversation_id=req.conversation_id,
        next_run_at=compute_next_run(req.schedule),
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    logger.info("Cron job %s created (%s)", job.id, job.schedule)
    return job


@router.get("", response_model=list[CronJobResponse])
async def list_cron_jobs(
    limit: int = 100,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_READ)),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(CronJob)
        .where(CronJob.organization_id == context.organization_id)
        .order_by(CronJob.created_at.desc())
        .limit(max(1, min(limit, 200)))
    )
    return result.scalars().all()


@router.patch("/{job_id}", response_model=CronJobResponse)
async def update_cron_job(
    job_id: str,
    req: CronJobUpdate,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
):
    job = await _get_owned_job(job_id, context, session)
    if req.schedule is not None:
        try:
            validate_schedule(req.schedule)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        job.schedule = req.schedule.strip()
        job.next_run_at = compute_next_run(job.schedule)
    if req.conversation_id is not None:
        await _check_conversation(req.conversation_id, context, session)
        job.conversation_id = req.conversation_id
    if req.name is not None:
        job.name = req.name
    if req.task is not None:
        job.task = req.task
    if req.max_steps is not None:
        job.max_steps = min(req.max_steps, context.policy.max_steps_per_run)
    if req.enabled is not None:
        job.enabled = req.enabled
        if req.enabled and job.next_run_at is None:
            job.next_run_at = compute_next_run(job.schedule)
    job.updated_at = datetime.now(timezone.utc)
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


@router.delete("/{job_id}", status_code=204)
async def delete_cron_job(
    job_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
):
    job = await _get_owned_job(job_id, context, session)
    await session.delete(job)
    await session.commit()
    return None
