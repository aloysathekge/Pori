"""Usage tracking and billing endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Date, cast, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ..database import get_session
from ..models import UsageRecord
from ..schemas import DailyUsageResponse, UsageRecordResponse, UsageSummaryResponse
from ..tenancy import OrganizationContext, Permission, require_permission

logger = logging.getLogger("aloy_backend")

router = APIRouter(prefix="/me/usage", tags=["usage"])


@router.get("", response_model=UsageSummaryResponse)
async def get_usage_summary(
    context: OrganizationContext = Depends(require_permission(Permission.USAGE_READ)),
    session: AsyncSession = Depends(get_session),
    days: int = Query(30, ge=1, le=365),
):
    """Get aggregated usage stats for the current user."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    result = await session.execute(
        select(UsageRecord)
        .where(UsageRecord.organization_id == context.organization_id)
        .where(UsageRecord.created_at >= since)
    )
    records = result.scalars().all()

    total_tokens = 0
    total_cost = 0.0
    by_model: dict = {}

    for r in records:
        total_tokens += r.total_tokens
        total_cost += r.estimated_cost

        key = f"{r.provider}/{r.model}" if r.provider else r.model
        if key not in by_model:
            by_model[key] = {"tokens": 0, "cost": 0.0, "requests": 0}
        by_model[key]["tokens"] += r.total_tokens
        by_model[key]["cost"] += r.estimated_cost
        by_model[key]["requests"] += 1

    return UsageSummaryResponse(
        total_tokens=total_tokens,
        total_cost=round(total_cost, 6),
        total_requests=len(records),
        by_model=by_model,
    )


@router.get("/history", response_model=list[DailyUsageResponse])
async def get_usage_history(
    context: OrganizationContext = Depends(require_permission(Permission.USAGE_READ)),
    session: AsyncSession = Depends(get_session),
    days: int = Query(30, ge=1, le=365),
):
    """Get daily usage breakdown."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    result = await session.execute(
        select(
            cast(UsageRecord.created_at, Date).label("date"),
            func.sum(UsageRecord.total_tokens).label("tokens"),
            func.sum(UsageRecord.estimated_cost).label("cost"),
            func.count().label("requests"),
        )
        .where(UsageRecord.organization_id == context.organization_id)
        .where(UsageRecord.created_at >= since)
        .group_by(cast(UsageRecord.created_at, Date))
        .order_by(cast(UsageRecord.created_at, Date))
    )

    return [
        DailyUsageResponse(
            date=str(row.date),
            tokens=int(row.tokens or 0),
            cost=round(float(row.cost or 0), 6),
            requests=int(row.requests or 0),
        )
        for row in result.all()
    ]


@router.get("/records", response_model=list[UsageRecordResponse])
async def list_usage_records(
    context: OrganizationContext = Depends(require_permission(Permission.USAGE_READ)),
    session: AsyncSession = Depends(get_session),
    limit: int = 50,
    offset: int = 0,
):
    """List individual usage records."""
    result = await session.execute(
        select(UsageRecord)
        .where(UsageRecord.organization_id == context.organization_id)
        .order_by(UsageRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return result.scalars().all()
