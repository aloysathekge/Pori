"""Published Event template discovery and idempotent installation endpoints."""

from __future__ import annotations

from typing import Any, NoReturn

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from ..database import get_session
from ..event_presenters import event_payload
from ..event_template_authoring import (
    TemplateReleaseImportInput,
    TemplateReleasePublishInput,
    ensure_catalog_operator,
    import_template_release,
    list_operator_releases,
    operator_release_payload,
    publish_template_release,
)
from ..event_templates import (
    EventTemplateError,
    find_template_installation,
    install_event_template,
    load_published_release,
)
from ..models import Event, EventTemplate, EventTemplateRelease, Run, SurfaceProject
from ..rate_limit import rate_limited_permission
from ..tenancy import OrganizationContext, Permission, require_permission

router = APIRouter(prefix="/event-templates", tags=["event-templates"])


class EventTemplateInstallBody(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    idempotency_key: str = Field(min_length=8, max_length=200)
    release_id: str | None = Field(default=None, min_length=1, max_length=200)
    title: str | None = Field(default=None, min_length=1, max_length=300)

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("idempotency_key must be trimmed")
        return value


def _catalog_payload(
    template: EventTemplate, release: EventTemplateRelease
) -> dict[str, Any]:
    return {
        "id": template.id,
        "slug": template.slug,
        "title": template.title,
        "summary": template.summary,
        "discovery_group": template.discovery_group,
        "current_release": {
            "id": release.id,
            "version": release.version,
            "schema_version": release.schema_version,
        },
        "updated_at": template.updated_at,
    }


def _raise_catalog_error(exc: EventTemplateError) -> NoReturn:
    raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.get("/operator/releases")
async def get_operator_template_releases(
    context: OrganizationContext = Depends(
        require_permission(Permission.OPERATOR_READ)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        ensure_catalog_operator(context)
        releases = await list_operator_releases(session)
    except EventTemplateError as exc:
        _raise_catalog_error(exc)
    return {"releases": releases}


@router.get("/operator/releases/{release_id}")
async def get_operator_template_release(
    release_id: str,
    context: OrganizationContext = Depends(
        require_permission(Permission.OPERATOR_READ)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        ensure_catalog_operator(context)
        release = await session.get(EventTemplateRelease, release_id)
        if release is None:
            raise EventTemplateError("Template release not found", status_code=404)
        return await operator_release_payload(session, release)
    except EventTemplateError as exc:
        _raise_catalog_error(exc)


@router.put("/operator/releases/imports", status_code=201)
async def import_operator_template_release(
    body: TemplateReleaseImportInput,
    context: OrganizationContext = Depends(require_permission(Permission.OPERATOR_ACT)),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        result = await import_template_release(session, context=context, body=body)
        await session.commit()
        return result
    except EventTemplateError as exc:
        await session.rollback()
        _raise_catalog_error(exc)
    except IntegrityError:
        await session.rollback()
        try:
            result = await import_template_release(session, context=context, body=body)
            await session.commit()
            return result
        except EventTemplateError as exc:
            await session.rollback()
            _raise_catalog_error(exc)
        except IntegrityError:
            await session.rollback()
        raise HTTPException(status_code=409, detail="Template import conflicted")


@router.post("/operator/releases/{release_id}/publish")
async def publish_operator_template_release(
    release_id: str,
    body: TemplateReleasePublishInput,
    context: OrganizationContext = Depends(require_permission(Permission.OPERATOR_ACT)),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        result = await publish_template_release(
            session,
            context=context,
            release_id=release_id,
            body=body,
        )
        await session.commit()
        return result
    except EventTemplateError as exc:
        await session.rollback()
        _raise_catalog_error(exc)
    except IntegrityError:
        await session.rollback()
        try:
            result = await publish_template_release(
                session,
                context=context,
                release_id=release_id,
                body=body,
            )
            await session.commit()
            return result
        except EventTemplateError as exc:
            await session.rollback()
            _raise_catalog_error(exc)
        except IntegrityError:
            await session.rollback()
        raise HTTPException(status_code=409, detail="Template publication conflicted")


@router.get("")
async def list_event_templates(
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """List valid published releases; taxonomy is not an entitlement check."""
    del context
    rows = (
        await session.execute(
            select(EventTemplate, EventTemplateRelease)
            .join(
                EventTemplateRelease,
                col(EventTemplateRelease.id) == col(EventTemplate.current_release_id),
            )
            .where(
                EventTemplate.status == "published",
                EventTemplateRelease.status == "published",
            )
            .order_by(
                col(EventTemplate.discovery_group),
                col(EventTemplate.title),
            )
        )
    ).all()
    templates: list[dict[str, Any]] = []
    for template, release in rows:
        try:
            await load_published_release(
                session,
                template_id=template.id,
                release_id=release.id,
            )
        except EventTemplateError:
            continue
        templates.append(_catalog_payload(template, release))
    return {"templates": templates}


@router.get("/{template_id}")
async def get_event_template(
    template_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    del context
    try:
        bundle = await load_published_release(session, template_id=template_id)
    except EventTemplateError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    payload = _catalog_payload(bundle.template, bundle.release)
    payload["release"] = {
        "id": bundle.release.id,
        "version": bundle.release.version,
        "schema_version": bundle.release.schema_version,
        "release_notes": bundle.release.release_notes,
        "checksum": bundle.release.checksum,
        "compatibility": [
            {
                "key": row.requirement_key,
                "requirement": row.requirement,
                "required": row.required,
            }
            for row in bundle.compatibility
        ],
        "assets": [
            {
                "key": row.asset_key,
                "kind": row.kind,
                "content_type": row.content_type,
                "sha256": row.sha256,
                "size_bytes": row.size_bytes,
            }
            for row in bundle.assets
        ],
        "guided_jobs": [
            {
                "key": row.job_key,
                "title": row.title,
                "priority": row.priority,
                "materializes_task": row.materialize_task,
            }
            for row in bundle.guided_jobs
        ],
    }
    return payload


@router.post("/{template_id}/install", status_code=201)
async def install_template(
    template_id: str,
    body: EventTemplateInstallBody,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.AGENT_WRITE)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        result = await install_event_template(
            session,
            organization_id=context.organization_id,
            user_id=context.user_id,
            template_id=template_id,
            release_id=body.release_id,
            idempotency_key=body.idempotency_key,
            title=body.title,
        )
        await session.commit()
    except EventTemplateError as exc:
        await session.rollback()
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except (ValueError, IntegrityError) as exc:
        await session.rollback()
        existing = await find_template_installation(
            session,
            organization_id=context.organization_id,
            user_id=context.user_id,
            idempotency_key=body.idempotency_key,
        )
        if existing is None:
            raise HTTPException(
                status_code=409,
                detail="Template installation failed validation",
            ) from exc
        try:
            result = await install_event_template(
                session,
                organization_id=context.organization_id,
                user_id=context.user_id,
                template_id=template_id,
                release_id=body.release_id,
                idempotency_key=body.idempotency_key,
                title=body.title,
            )
            await session.commit()
        except EventTemplateError as replay_exc:
            raise HTTPException(
                status_code=replay_exc.status_code,
                detail=replay_exc.detail,
            ) from replay_exc

    event = await session.get(Event, result.event.id)
    if event is None:
        raise HTTPException(status_code=409, detail="Installed Event is unavailable")
    project = (
        (
            await session.execute(
                select(SurfaceProject).where(
                    SurfaceProject.organization_id == context.organization_id,
                    SurfaceProject.user_id == context.user_id,
                    SurfaceProject.event_id == event.id,
                )
            )
        )
        .scalars()
        .first()
    )
    surface_run = (
        await session.get(Run, result.surface_run_id)
        if result.surface_run_id is not None
        else None
    )
    surface_status = "not_seeded"
    if project is not None:
        if project.published_build_id and project.published_revision_id:
            surface_status = "published"
        elif surface_run is not None and surface_run.status in {"pending", "running"}:
            surface_status = "preparing"
        elif surface_run is not None and surface_run.status in {"failed", "cancelled"}:
            surface_status = "failed"
        else:
            surface_status = "source_seeded"
    return {
        "installation": {
            "id": result.installation.id,
            "template_id": result.installation.template_id,
            "release_id": result.installation.release_id,
            "event_id": result.installation.event_id,
            "status": result.installation.status,
            "installed_at": result.installation.installed_at,
        },
        "event": event_payload(event),
        "surface": {
            "project_id": project.id if project is not None else None,
            "status": surface_status,
            "run_id": surface_run.id if surface_run is not None else None,
            "run_status": surface_run.status if surface_run is not None else None,
        },
        "replayed": result.replayed,
    }


__all__ = ["router"]
