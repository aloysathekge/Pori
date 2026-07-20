"""Atomic publication and last-good rollback for Event Surfaces."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select, update

from .models import (
    Event,
    EventTrailEntry,
    SurfaceBuild,
    SurfaceProject,
    SurfacePublication,
    SurfaceRevision,
)
from .storage import ObjectStore
from .surface_authoring import SurfaceAuthoringError, SurfaceConflictError
from .surface_quality import surface_quality_receipt_error
from .surface_runtime import InvalidSurfaceBundle, build_surface_runtime_document


class SurfacePublicationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    build_id: str = Field(min_length=1, max_length=200)
    expected_published_revision_id: str | None = Field(default=None, max_length=200)
    expected_published_build_id: str | None = Field(default=None, max_length=200)
    idempotency_key: str = Field(min_length=8, max_length=200)

    @field_validator(
        "build_id",
        "expected_published_revision_id",
        "expected_published_build_id",
        "idempotency_key",
    )
    @classmethod
    def validate_trimmed(cls, value: str | None) -> str | None:
        if value is not None and value != value.strip():
            raise ValueError("Surface publication identifiers must be trimmed")
        return value


def surface_publication_payload(
    publication: SurfacePublication,
    *,
    replayed: bool = False,
) -> dict[str, Any]:
    return {
        "id": publication.id,
        "event_id": publication.event_id,
        "project_id": publication.project_id,
        "revision_id": publication.revision_id,
        "revision_number": publication.revision_number,
        "build_id": publication.build_id,
        "previous_revision_id": publication.previous_revision_id,
        "previous_build_id": publication.previous_build_id,
        "action": publication.action,
        "actor_id": publication.actor_id,
        "run_id": publication.run_id,
        "created_at": publication.created_at,
        "replayed": replayed,
    }


def _request_fingerprint(
    action: Literal["publish", "rollback"],
    params: SurfacePublicationParams,
) -> str:
    encoded = json.dumps(
        {
            "action": action,
            "build_id": params.build_id,
            "expected_published_revision_id": params.expected_published_revision_id,
            "expected_published_build_id": params.expected_published_build_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


async def _verify_artifact(
    build: SurfaceBuild,
    object_store: ObjectStore,
    *,
    require_quality: bool,
) -> None:
    if (
        build.status != "succeeded"
        or build.validation_result.get("passed") is not True
        or not build.bundle_key
        or not build.bundle_sha256
    ):
        raise SurfaceAuthoringError(
            "Only a successful, validated Surface build can be published"
        )
    if require_quality:
        quality_error = surface_quality_receipt_error(build)
        if quality_error is not None:
            raise SurfaceAuthoringError(
                "Surface publication quality gate failed: " + quality_error
            )

    def read_bundle() -> bytes:
        with object_store.open(build.bundle_key or "") as stream:
            return stream.read()

    try:
        bundle = await asyncio.to_thread(read_bundle)
    except Exception as exc:
        raise SurfaceAuthoringError(
            "Surface build artifact is unavailable; the live Surface was not changed"
        ) from exc
    if hashlib.sha256(bundle).hexdigest() != build.bundle_sha256:
        raise SurfaceAuthoringError(
            "Surface build artifact checksum failed; the live Surface was not changed"
        )
    try:
        build_surface_runtime_document(bundle)
    except InvalidSurfaceBundle as exc:
        raise SurfaceAuthoringError(
            "Surface build artifact is invalid; the live Surface was not changed"
        ) from exc


async def published_surface_snapshot(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    event_id: str,
) -> dict[str, Any]:
    result = await session.execute(
        select(SurfaceProject).where(
            SurfaceProject.organization_id == organization_id,
            SurfaceProject.user_id == user_id,
            SurfaceProject.event_id == event_id,
        )
    )
    project = result.scalars().first()
    if project is None or (
        project.published_revision_id is None and project.published_build_id is None
    ):
        return {"project_id": project.id if project else None, "build": None}
    if not project.published_revision_id or not project.published_build_id:
        raise SurfaceAuthoringError("Published Surface pointer is inconsistent")
    build_result = await session.execute(
        select(SurfaceBuild).where(
            SurfaceBuild.id == project.published_build_id,
            SurfaceBuild.revision_id == project.published_revision_id,
            SurfaceBuild.project_id == project.id,
            SurfaceBuild.organization_id == organization_id,
            SurfaceBuild.user_id == user_id,
            SurfaceBuild.event_id == event_id,
            SurfaceBuild.status == "succeeded",
            col(SurfaceBuild.bundle_key).is_not(None),
        )
    )
    build = build_result.scalars().first()
    if build is None:
        raise SurfaceAuthoringError("Published Surface build is unavailable")
    return {
        "project_id": project.id,
        "published_revision_id": project.published_revision_id,
        "published_build_id": project.published_build_id,
        "build": build,
    }


async def list_surface_publications(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    event_id: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    rows = list(
        (
            await session.execute(
                select(SurfacePublication)
                .where(
                    SurfacePublication.organization_id == organization_id,
                    SurfacePublication.user_id == user_id,
                    SurfacePublication.event_id == event_id,
                )
                .order_by(col(SurfacePublication.created_at).desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return [surface_publication_payload(row) for row in rows]


async def change_surface_publication(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    event_id: str,
    actor_id: str,
    run_id: str | None,
    params: SurfacePublicationParams,
    action: Literal["publish", "rollback"],
    object_store: ObjectStore,
) -> dict[str, Any]:
    event = await session.get(Event, event_id)
    if (
        event is None
        or event.organization_id != organization_id
        or event.user_id != user_id
    ):
        raise SurfaceAuthoringError("Event is unavailable")
    project_result = await session.execute(
        select(SurfaceProject).where(
            SurfaceProject.organization_id == organization_id,
            SurfaceProject.user_id == user_id,
            SurfaceProject.event_id == event.id,
        )
    )
    project = project_result.scalars().first()
    if project is None:
        raise SurfaceAuthoringError("Surface project is unavailable")
    project_id = project.id

    fingerprint = _request_fingerprint(action, params)
    replay_result = await session.execute(
        select(SurfacePublication).where(
            SurfacePublication.project_id == project_id,
            SurfacePublication.idempotency_key == params.idempotency_key,
        )
    )
    replay = replay_result.scalars().first()
    if replay is not None:
        if replay.request_fingerprint != fingerprint:
            raise SurfaceConflictError(
                "idempotency_key was already used for a different publication"
            )
        return surface_publication_payload(replay, replayed=True)

    target_result = await session.execute(
        select(SurfaceBuild, SurfaceRevision)
        .join(SurfaceRevision, col(SurfaceRevision.id) == col(SurfaceBuild.revision_id))
        .where(
            SurfaceBuild.id == params.build_id,
            SurfaceBuild.project_id == project.id,
            SurfaceBuild.organization_id == organization_id,
            SurfaceBuild.user_id == user_id,
            SurfaceBuild.event_id == event.id,
            SurfaceRevision.project_id == project.id,
            SurfaceRevision.organization_id == organization_id,
            SurfaceRevision.user_id == user_id,
            SurfaceRevision.event_id == event.id,
        )
    )
    target = target_result.first()
    if target is None:
        raise SurfaceAuthoringError("Surface build is unavailable")
    build, revision = target
    if action == "publish" and project.draft_revision_id != revision.id:
        raise SurfaceConflictError(
            "Surface draft changed after this build; build the current draft before publishing"
        )
    if action == "rollback":
        previous_result = await session.execute(
            select(SurfacePublication.id).where(
                SurfacePublication.project_id == project_id,
                SurfacePublication.build_id == build.id,
            )
        )
        if previous_result.first() is None:
            raise SurfaceAuthoringError(
                "Rollback target was never a published last-good Surface"
            )
        if project.published_build_id == build.id:
            raise SurfaceConflictError("Surface build is already published")

    if (
        project.published_revision_id != params.expected_published_revision_id
        or project.published_build_id != params.expected_published_build_id
    ):
        raise SurfaceConflictError(
            "Published Surface changed; read the project and retry against the current pointer"
        )
    # A new publication must carry a passing receipt bound to this exact build.
    # Rollback remains available for a previously published legacy last-good
    # build so introducing a stricter policy cannot remove recovery authority.
    await _verify_artifact(
        build,
        object_store,
        require_quality=action == "publish",
    )

    publication = SurfacePublication(
        organization_id=organization_id,
        user_id=user_id,
        event_id=event.id,
        project_id=project.id,
        revision_id=revision.id,
        revision_number=revision.revision_number,
        build_id=build.id,
        previous_revision_id=project.published_revision_id,
        previous_build_id=project.published_build_id,
        action=action,
        actor_id=actor_id,
        run_id=run_id,
        idempotency_key=params.idempotency_key,
        request_fingerprint=fingerprint,
    )
    session.add(publication)
    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        concurrent_result = await session.execute(
            select(SurfacePublication).where(
                SurfacePublication.project_id == project_id,
                SurfacePublication.idempotency_key == params.idempotency_key,
            )
        )
        concurrent = concurrent_result.scalars().first()
        if concurrent is not None and concurrent.request_fingerprint == fingerprint:
            return surface_publication_payload(concurrent, replayed=True)
        raise SurfaceConflictError(
            "Surface publication was created concurrently; read the project and retry"
        ) from exc

    expected_revision = (
        col(SurfaceProject.published_revision_id).is_(None)
        if params.expected_published_revision_id is None
        else col(SurfaceProject.published_revision_id)
        == params.expected_published_revision_id
    )
    expected_build = (
        col(SurfaceProject.published_build_id).is_(None)
        if params.expected_published_build_id is None
        else col(SurfaceProject.published_build_id)
        == params.expected_published_build_id
    )
    now = datetime.now(timezone.utc)
    claimed = await session.execute(
        update(SurfaceProject)
        .where(
            col(SurfaceProject.id) == project.id,
            col(SurfaceProject.organization_id) == organization_id,
            col(SurfaceProject.user_id) == user_id,
            col(SurfaceProject.event_id) == event.id,
            expected_revision,
            expected_build,
        )
        .values(
            published_revision_id=revision.id,
            published_build_id=build.id,
            lifecycle="published",
            updated_at=now,
        )
    )
    if claimed.rowcount != 1:  # type: ignore[attr-defined]
        await session.rollback()
        raise SurfaceConflictError(
            "Published Surface changed concurrently; read the project and retry"
        )

    event.updated_at = now
    session.add(event)
    session.add(
        EventTrailEntry(
            organization_id=organization_id,
            user_id=user_id,
            event_id=event.id,
            actor_id=actor_id,
            kind="surface_published" if action == "publish" else "surface_rolled_back",
            summary=(
                f"Published Surface revision {revision.revision_number}"
                if action == "publish"
                else f"Restored Surface revision {revision.revision_number}"
            ),
            run_id=run_id,
            evidence_refs=[
                {"surface_revision_id": revision.id},
                {"surface_build_id": build.id},
                {"bundle_sha256": build.bundle_sha256},
            ],
            payload={
                "project_id": project.id,
                "revision_id": revision.id,
                "build_id": build.id,
                "previous_revision_id": publication.previous_revision_id,
                "previous_build_id": publication.previous_build_id,
                "action": action,
            },
        )
    )
    await session.commit()
    return surface_publication_payload(publication)


__all__ = [
    "SurfacePublicationParams",
    "change_surface_publication",
    "list_surface_publications",
    "published_surface_snapshot",
    "surface_publication_payload",
]
