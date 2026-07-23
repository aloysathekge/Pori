"""Deliver the bundled baseline Surface to every custom Event.

``docs/aloy-baseline-surface-spec.md`` (S3): a new custom Event immediately
receives a persisted draft of the reviewed baseline source and a queued
model-free materialization Run. The ordinary host build, inspection, and
publication gate makes it live in the background — Event creation never
waits, and a delivery failure never blocks the Event. Life is exempt, and
Events that already own a Surface project (template installs, replays) are
left untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from .baseline_surface import baseline_surface_files, baseline_surface_fingerprint
from .config import settings
from .models import Event, Organization, Run, SurfaceProject, SurfaceRevision
from .surface_manifest import parse_surface_manifest
from .surface_materialization import (
    SURFACE_MATERIALIZATION_RUN_KIND,
    queue_surface_revision_materialization,
)
from .tenancy import OrganizationPolicy

logger = logging.getLogger("aloy_backend.baseline_delivery")

BASELINE_SURFACE_RELEASE = "aloy-baseline-surface@1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


async def ensure_event_baseline_surface(
    session: AsyncSession,
    *,
    event: Event,
    actor_id: str,
) -> str | None:
    """Persist the baseline draft and queue its materialization, idempotently.

    Returns the materialization Run id, or ``None`` when the Event is exempt
    or already owns Surface state. The caller commits.
    """
    if event.is_life or event.lifecycle == "archived":
        return None
    existing_run = (
        (
            await session.execute(
                select(Run).where(
                    Run.organization_id == event.organization_id,
                    Run.user_id == event.user_id,
                    Run.event_id == event.id,
                    Run.run_kind == SURFACE_MATERIALIZATION_RUN_KIND,
                )
            )
        )
        .scalars()
        .first()
    )
    if existing_run is not None:
        return existing_run.id
    existing_project = (
        (
            await session.execute(
                select(SurfaceProject).where(
                    SurfaceProject.organization_id == event.organization_id,
                    SurfaceProject.user_id == event.user_id,
                    SurfaceProject.event_id == event.id,
                )
            )
        )
        .scalars()
        .first()
    )
    if existing_project is not None:
        return None

    files = baseline_surface_files()
    manifest = parse_surface_manifest(files).model_dump(mode="json", by_alias=True)
    source_checksum = hashlib.sha256(
        _canonical_bytes({"manifest": manifest, "files": files})
    ).hexdigest()
    project = SurfaceProject(
        organization_id=event.organization_id,
        user_id=event.user_id,
        event_id=event.id,
        sdk_version=str(manifest.get("sdk_version") or "1"),
        data_revision=0,
        lifecycle="draft",
    )
    session.add(project)
    await session.flush()
    revision = SurfaceRevision(
        organization_id=event.organization_id,
        user_id=event.user_id,
        event_id=event.id,
        project_id=project.id,
        revision_number=1,
        idempotency_key=f"baseline:{event.id}",
        request_fingerprint=baseline_surface_fingerprint(),
        manifest=manifest,
        files=dict(files),
        checksum=source_checksum,
        file_count=len(files),
        total_bytes=sum(len(value.encode("utf-8")) for value in files.values()),
    )
    session.add(revision)
    await session.flush()
    project.draft_revision_id = revision.id
    session.add(project)

    organization = await session.get(Organization, event.organization_id)
    policy = OrganizationPolicy.model_validate(
        organization.policy if organization is not None else {}
    )
    run, _ = await queue_surface_revision_materialization(
        session,
        event=event,
        project=project,
        revision=revision,
        policy=policy,
        trigger="event_baseline",
        actor_id=actor_id,
        origin_evidence=[
            {"baseline_release": BASELINE_SURFACE_RELEASE},
            {"baseline_fingerprint": baseline_surface_fingerprint()},
        ],
    )
    return run.id


async def deliver_event_baseline_surface(
    session: AsyncSession,
    *,
    event: Event,
    actor_id: str,
) -> str | None:
    """Fail-safe wrapper for the Event-creation path.

    The baseline is a background enrichment: any delivery failure is logged
    and the Event is created normally without a Surface (the ordinary Surface
    request path still works later).
    """
    if not settings.surface_baseline_enabled:
        return None
    try:
        return await ensure_event_baseline_surface(
            session, event=event, actor_id=actor_id
        )
    except Exception:
        logger.exception(
            "Baseline Surface delivery failed for Event %s; continuing", event.id
        )
        return None


__all__ = [
    "BASELINE_SURFACE_RELEASE",
    "deliver_event_baseline_surface",
    "ensure_event_baseline_surface",
]
