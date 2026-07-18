"""Tenant-scoped host reads for model-authored Event Surface metadata.

Source mutation intentionally does not have a public route. Models author
through the authenticated internal product-tool boundary.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from ..database import get_session
from ..models import Event, Run, SurfaceBuild, SurfaceProject
from ..rate_limit import rate_limited_permission
from ..storage import get_object_store
from ..surface_authoring import (
    SurfaceAuthoringError,
    SurfaceConflictError,
    surface_project_snapshot,
)
from ..surface_builds import surface_build_payload
from ..surface_interactions import (
    SurfaceInteractionError,
    SurfaceInteractionRequest,
    handle_surface_interaction,
    record_surface_interaction_rejection,
    surface_runtime_context,
)
from ..surface_publication import (
    SurfacePublicationParams,
    change_surface_publication,
    list_surface_publications,
    published_surface_snapshot,
)
from ..surface_runtime import InvalidSurfaceBundle, build_surface_runtime_document
from ..tenancy import OrganizationContext, Permission

router = APIRouter(prefix="/events/{event_id}/surface", tags=["surfaces"])
logger = logging.getLogger("aloy_backend.routes.surfaces")

SURFACE_BUILDER_RUN_KIND = "surface_builder"

_SURFACE_STAGE_MESSAGES = {
    "generating_candidate": "Designing and writing your Surface",
    "validating_candidate": "Checking the generated application",
    "building_bundle": "Compiling the Surface",
    "inspecting_preview": "Checking that the Surface opens correctly",
    "publishing_surface": "Publishing the new Surface",
    "repairing_candidate": "Repairing the Surface",
}


def _surface_error(exc: SurfaceInteractionError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={
            "message": exc.detail,
            "code": exc.code,
            "retryable": exc.retryable,
            "attempt_id": exc.attempt_id,
        },
    )


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


def _aware(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def _surface_activity_payload(run: Run) -> dict:
    now = datetime.now(timezone.utc)
    progress = dict(run.progress or {})
    stage = str(progress.get("stage") or "queued")
    lease_expires_at = _aware(run.lease_expires_at)
    overdue = (
        run.status == "running"
        and lease_expires_at is not None
        and lease_expires_at < now
    )
    status = "overdue" if overdue else run.status
    if status == "pending":
        message = "Waiting for the Surface Builder"
    elif status == "completed" and run.success:
        message = "Your Surface is ready"
    elif status == "overdue":
        message = "The Surface Builder stopped reporting progress"
    elif status in {"failed", "cancelled"}:
        message = "The Surface could not be completed"
    else:
        message = _SURFACE_STAGE_MESSAGES.get(stage, "Building your Surface")
    started_at = _aware(run.started_at or run.created_at)
    return {
        "run_id": run.id,
        "status": status,
        "stage": stage,
        "message": message,
        "submission": int(progress.get("submission") or 1),
        "attempt_count": run.attempt_count,
        "max_attempts": run.max_attempts,
        "started_at": started_at,
        "updated_at": progress.get("updated_at") or started_at,
        "completed_at": _aware(run.completed_at),
        "elapsed_seconds": (
            max(0, int((now - started_at).total_seconds())) if started_at else 0
        ),
        "active": status in {"pending", "running"},
    }


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


@router.get("/status")
async def get_surface_activity(
    event_id: str,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_READ)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict | None:
    """Return the latest durable Builder state, including pre-build work."""
    event = await _owned_event(session, context, event_id)
    run = (
        (
            await session.execute(
                select(Run)
                .where(
                    Run.organization_id == context.organization_id,
                    Run.user_id == context.user_id,
                    Run.event_id == event.id,
                    Run.run_kind == SURFACE_BUILDER_RUN_KIND,
                )
                .order_by(col(Run.created_at).desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    return _surface_activity_payload(run) if run is not None else None


@router.get("/runtime")
async def get_published_surface_runtime(
    event_id: str,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_READ)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    event = await _owned_event(session, context, event_id)
    try:
        snapshot = await published_surface_snapshot(
            session,
            organization_id=context.organization_id,
            user_id=context.user_id,
            event_id=event.id,
        )
    except SurfaceAuthoringError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    build = snapshot.get("build")
    return {
        "project_id": snapshot.get("project_id"),
        "published_revision_id": snapshot.get("published_revision_id"),
        "published_build_id": snapshot.get("published_build_id"),
        "build": (
            surface_build_payload(build, include_log=False)
            if build is not None
            else None
        ),
    }


@router.get("/publications")
async def get_surface_publications(
    event_id: str,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_READ)
    ),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    event = await _owned_event(session, context, event_id)
    return await list_surface_publications(
        session,
        organization_id=context.organization_id,
        user_id=context.user_id,
        event_id=event.id,
    )


async def _rollback_from_route(
    *,
    event: Event,
    body: SurfacePublicationParams,
    context: OrganizationContext,
    session: AsyncSession,
) -> dict:
    try:
        return await change_surface_publication(
            session,
            organization_id=context.organization_id,
            user_id=context.user_id,
            event_id=event.id,
            actor_id=context.user_id,
            run_id=None,
            params=body,
            action="rollback",
            object_store=get_object_store(),
        )
    except SurfaceConflictError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SurfaceAuthoringError as exc:
        await session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/rollback")
async def rollback_surface(
    event_id: str,
    body: SurfacePublicationParams,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_CREATE)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    event = await _owned_event(session, context, event_id)
    return await _rollback_from_route(
        event=event,
        body=body,
        context=context,
        session=session,
    )


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


@router.get("/builds/{build_id}/runtime-document")
async def get_surface_runtime_document(
    event_id: str,
    build_id: str,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_READ)
    ),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Return an authenticated preview document for one immutable build.

    The object-store key and bundle are never exposed. The host validates the
    fixed bundle contract and supplies all HTML and security policy.
    """
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
    if build.status != "succeeded" or not build.bundle_key:
        raise HTTPException(status_code=409, detail="Surface build is not renderable")

    def read_bundle() -> bytes:
        with get_object_store().open(build.bundle_key or "") as stream:
            return stream.read()

    try:
        bundle = await asyncio.to_thread(read_bundle)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=409, detail="Surface build artifact is unavailable"
        ) from exc
    try:
        document = build_surface_runtime_document(bundle)
    except InvalidSurfaceBundle as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return Response(
        content=document.html,
        media_type="text/html",
        headers={
            "Cache-Control": "private, no-store",
            "Content-Security-Policy": document.content_security_policy,
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/context")
async def get_surface_runtime_context(
    event_id: str,
    build_id: str = Query(min_length=1, max_length=200),
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_READ)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Return only the capabilities declared by one immutable build."""
    try:
        return await surface_runtime_context(
            session,
            context=context,
            event_id=event_id,
            build_id=build_id,
        )
    except SurfaceInteractionError as exc:
        raise _surface_error(exc) from exc


@router.post("/interactions", status_code=202)
async def create_surface_interaction(
    event_id: str,
    body: SurfaceInteractionRequest,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_CREATE)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Validate and persist a request from the bound iframe bridge."""
    try:
        return await handle_surface_interaction(
            session,
            context=context,
            event_id=event_id,
            request=body,
        )
    except SurfaceInteractionError as exc:
        await session.rollback()
        try:
            await record_surface_interaction_rejection(
                session,
                context=context,
                event_id=event_id,
                request=body,
                error=exc,
            )
        except Exception:
            await session.rollback()
            logger.exception("Failed to persist rejected Surface command attempt")
        raise _surface_error(exc) from exc


__all__ = ["router"]
