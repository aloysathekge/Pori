"""Tenant-scoped host reads for model-authored Event Surface metadata.

Source mutation intentionally does not have a public route. Models author
through the authenticated internal product-tool boundary.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from ..database import get_session
from ..models import Event, SurfaceBuild, SurfaceProject
from ..rate_limit import rate_limited_permission
from ..surface_authoring import surface_project_snapshot
from ..surface_builds import surface_build_payload
from ..tenancy import OrganizationContext, Permission

router = APIRouter(prefix="/events/{event_id}/surface", tags=["surfaces"])


async def _owned_event(
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


@router.get("/project")
async def get_surface_project(
    event_id: str,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_READ)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    event = await _owned_event(session, context, event_id)
    return await surface_project_snapshot(
        session,
        organization_id=context.organization_id,
        user_id=context.user_id,
        event_id=event.id,
        include_files=False,
    )


@router.get("/builds")
async def list_surface_builds(
    event_id: str,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_READ)
    ),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    event = await _owned_event(session, context, event_id)
    rows = list(
        (
            await session.execute(
                select(SurfaceBuild)
                .where(
                    SurfaceBuild.organization_id == context.organization_id,
                    SurfaceBuild.user_id == context.user_id,
                    SurfaceBuild.event_id == event.id,
                )
                .order_by(col(SurfaceBuild.created_at).desc())
                .limit(50)
            )
        )
        .scalars()
        .all()
    )
    return [surface_build_payload(build, include_log=False) for build in rows]


@router.get("/builds/{build_id}")
async def get_surface_build(
    event_id: str,
    build_id: str,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_READ)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    event = await _owned_event(session, context, event_id)
    result = await session.execute(
        select(SurfaceBuild)
        .join(SurfaceProject, col(SurfaceProject.id) == col(SurfaceBuild.project_id))
        .where(
            SurfaceBuild.id == build_id,
            SurfaceBuild.organization_id == context.organization_id,
            SurfaceBuild.user_id == context.user_id,
            SurfaceBuild.event_id == event.id,
            SurfaceProject.organization_id == context.organization_id,
            SurfaceProject.user_id == context.user_id,
            SurfaceProject.event_id == event.id,
        )
    )
    build = result.scalars().first()
    if build is None:
        raise HTTPException(status_code=404, detail="Surface build not found")
    return surface_build_payload(build, include_log=False)


__all__ = ["router"]
