"""Tenant-scoped host reads for model-authored Event Surface metadata.

Source mutation intentionally does not have a public route. Models author
through the authenticated internal product-tool boundary.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models import Event
from ..rate_limit import rate_limited_permission
from ..surface_authoring import surface_project_snapshot
from ..tenancy import OrganizationContext, Permission

router = APIRouter(prefix="/events/{event_id}/surface", tags=["surfaces"])


@router.get("/project")
async def get_surface_project(
    event_id: str,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_READ)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    event = await session.get(Event, event_id)
    if (
        event is None
        or event.organization_id != context.organization_id
        or event.user_id != context.user_id
    ):
        raise HTTPException(status_code=404, detail="Event not found")
    return await surface_project_snapshot(
        session,
        organization_id=context.organization_id,
        user_id=context.user_id,
        event_id=event.id,
        include_files=False,
    )


__all__ = ["router"]
