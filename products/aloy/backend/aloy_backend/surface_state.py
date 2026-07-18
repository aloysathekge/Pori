"""Canonical Event-scoped Surface state projection and detailed reads."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from .models import SurfaceDataRecord, SurfaceProject
from .surface_commands import SURFACE_COMMAND_CONTRACT_VERSION

MAX_CONTEXT_RECORDS = 100
MAX_CONTEXT_CHARS = 32_000
MAX_DETAILED_RECORDS = 500


def surface_record_payload(record: SurfaceDataRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "namespace": record.namespace,
        "key": record.record_key,
        "data": record.data,
        "revision": record.revision,
        "posture": record.posture,
        "actor_id": record.actor_id,
        "provenance": record.provenance,
        "evidence_refs": record.evidence_refs,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }


async def _event_project(
    session: AsyncSession, *, organization_id: str, user_id: str, event_id: str
) -> SurfaceProject | None:
    return (
        (
            await session.execute(
                select(SurfaceProject).where(
                    SurfaceProject.organization_id == organization_id,
                    SurfaceProject.user_id == user_id,
                    SurfaceProject.event_id == event_id,
                )
            )
        )
        .scalars()
        .first()
    )


async def read_surface_state(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    event_id: str,
    namespace: str | None = None,
    keys: list[str] | None = None,
    limit: int = MAX_DETAILED_RECORDS,
) -> dict[str, Any]:
    """Read canonical records only after tenant and Event scope are fixed."""
    project = await _event_project(
        session,
        organization_id=organization_id,
        user_id=user_id,
        event_id=event_id,
    )
    if project is None:
        return {
            "contract_version": SURFACE_COMMAND_CONTRACT_VERSION,
            "event_id": event_id,
            "project_id": None,
            "data_revision": 0,
            "records": [],
        }
    statement = select(SurfaceDataRecord).where(
        SurfaceDataRecord.organization_id == organization_id,
        SurfaceDataRecord.user_id == user_id,
        SurfaceDataRecord.event_id == event_id,
        SurfaceDataRecord.project_id == project.id,
    )
    if namespace is not None:
        statement = statement.where(SurfaceDataRecord.namespace == namespace)
    normalized_keys = [value.strip()[:200] for value in keys or [] if value.strip()]
    if normalized_keys:
        statement = statement.where(
            col(SurfaceDataRecord.record_key).in_(normalized_keys)
        )
    records = list(
        (
            await session.execute(
                statement.order_by(
                    col(SurfaceDataRecord.namespace),
                    col(SurfaceDataRecord.record_key),
                ).limit(max(1, min(limit, MAX_DETAILED_RECORDS)))
            )
        )
        .scalars()
        .all()
    )
    return {
        "contract_version": SURFACE_COMMAND_CONTRACT_VERSION,
        "event_id": event_id,
        "project_id": project.id,
        "data_revision": project.data_revision,
        "records": [surface_record_payload(record) for record in records],
    }


async def surface_state_context_projection(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    event_id: str,
) -> dict[str, Any] | None:
    """Return a deterministic bounded projection for the Event context prefix."""
    detailed = await read_surface_state(
        session,
        organization_id=organization_id,
        user_id=user_id,
        event_id=event_id,
        limit=MAX_CONTEXT_RECORDS,
    )
    if detailed["project_id"] is None:
        return None
    accepted: list[dict[str, Any]] = []
    used = 0
    for record in detailed["records"]:
        compact = {
            "namespace": record["namespace"],
            "key": record["key"],
            "data": record["data"],
            "revision": record["revision"],
            "posture": record["posture"],
        }
        size = len(json.dumps(compact, sort_keys=True, separators=(",", ":")))
        if accepted and used + size > MAX_CONTEXT_CHARS:
            break
        accepted.append(compact)
        used += size
    return {
        "contract_version": detailed["contract_version"],
        "project_id": detailed["project_id"],
        "data_revision": detailed["data_revision"],
        "records": accepted,
        "record_count": len(detailed["records"]),
        "projection_truncated": len(accepted) < len(detailed["records"]),
    }


__all__ = [
    "read_surface_state",
    "surface_record_payload",
    "surface_state_context_projection",
]
