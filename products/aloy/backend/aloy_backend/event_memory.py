"""Event-scoped memory inspection and user-controlled mutations.

This module owns the product invariants for Event memory.  Routes provide
authorization; this layer preserves scope, provenance, supersession history,
Trail evidence, and Event-context cache invalidation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from pori import ConflictPolicy, MemoryCatalog, MemoryStatus

from .event_bootstrap import queue_event_bootstrap_if_ready
from .event_context import refresh_event_context_snapshot
from .memory_records import create_record, record_to_row, row_to_record
from .models import Event, EventBrief, EventTrailEntry, KnowledgeEntry
from .scope_resolver import ORG, resolve_layered

_CANONICAL_RECORD_TYPES = frozenset({"web_evidence", "event_record", "research_report"})


def _is_user_managed_memory(row: KnowledgeEntry) -> bool:
    return (row.metadata_ or {}).get("record_type") not in _CANONICAL_RECORD_TYPES


@dataclass(frozen=True)
class EventMemoryMutationError(Exception):
    """Safe domain error returned by Event-memory mutation endpoints."""

    detail: str
    status_code: int = 409

    def __str__(self) -> str:
        return self.detail


async def list_event_memory(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    event_id: str,
    event_limit: int = 100,
    global_limit: int = 25,
) -> tuple[list[KnowledgeEntry], list[KnowledgeEntry], int, int, set[str]]:
    """Return active Event memory and inherited global memory separately."""

    accessible_owner = or_(
        col(KnowledgeEntry.user_id) == user_id,
        col(KnowledgeEntry.scope_level) == ORG,
    )
    event_result = await session.execute(
        select(KnowledgeEntry)
        .where(
            KnowledgeEntry.organization_id == organization_id,
            accessible_owner,
            KnowledgeEntry.event_id == event_id,
            KnowledgeEntry.status == MemoryStatus.ACTIVE.value,
        )
        .order_by(col(KnowledgeEntry.updated_at).desc())
    )
    global_result = await session.execute(
        select(KnowledgeEntry)
        .where(
            KnowledgeEntry.organization_id == organization_id,
            accessible_owner,
            col(KnowledgeEntry.event_id).is_(None),
            KnowledgeEntry.status == MemoryStatus.ACTIVE.value,
        )
        .order_by(col(KnowledgeEntry.updated_at).desc())
    )
    event_rows = [
        row
        for row in resolve_layered(event_result.scalars().all(), event_id=event_id)
        if _is_user_managed_memory(row)
    ]
    global_rows = [
        row
        for row in resolve_layered(global_result.scalars().all())
        if _is_user_managed_memory(row)
    ]
    return (
        event_rows[:event_limit],
        global_rows[:global_limit],
        len(event_rows),
        len(global_rows),
        {row.id for row in global_rows},
    )


async def correct_event_memory(
    session: AsyncSession,
    *,
    event: Event,
    organization_id: str,
    user_id: str,
    memory_id: str,
    content: str,
    reason: str | None = None,
) -> KnowledgeEntry:
    """Supersede one Event memory with a user-authored correction."""

    source = await _load_mutable_event_memory(
        session,
        organization_id=organization_id,
        user_id=user_id,
        event_id=event.id,
        memory_id=memory_id,
    )
    normalized = content.strip()
    if normalized == source.content.strip():
        raise EventMemoryMutationError("The correction must change the memory")

    source_metadata = dict(source.metadata_ or {})
    source_metadata.pop("promoted_global_id", None)
    source_metadata.pop("promoted_at", None)
    source_metadata["corrected_from"] = source.id
    if reason:
        source_metadata["correction_reason"] = reason.strip()

    corrected = create_record(
        user_id=user_id,
        organization_id=organization_id,
        event_id=event.id,
        content=normalized,
        tags=list(source.tags or []),
        importance=source.importance,
        kind=source.kind,
        confidence=1.0,
        sensitivity=source.sensitivity,
        source="user_correction",
        source_id=source.id,
        retention=dict(source.retention or {}),
        conflict_key=source.conflict_key or f"event-memory:{source.id}",
        event_at=source.event_at,
        metadata=source_metadata,
    )
    now = datetime.now(timezone.utc)
    source.status = MemoryStatus.SUPERSEDED.value
    source.superseded_by = corrected.id
    source.updated_at = now
    corrected_row = record_to_row(corrected)
    session.add(source)
    session.add(corrected_row)
    session.add(
        EventTrailEntry(
            organization_id=organization_id,
            user_id=user_id,
            event_id=event.id,
            actor_id=user_id,
            kind="event_memory_corrected",
            summary="Corrected an Event memory",
            evidence_refs=[
                {"knowledge_entry_id": source.id},
                {"replacement_knowledge_entry_id": corrected.id},
            ],
            payload={
                "kind": corrected.kind.value,
                "sensitivity": corrected.sensitivity.value,
            },
        )
    )
    event.updated_at = now
    session.add(event)
    await session.flush()
    await _refresh_event_context(session, event=event)
    return corrected_row


async def forget_event_memory(
    session: AsyncSession,
    *,
    event: Event,
    organization_id: str,
    user_id: str,
    memory_id: str,
) -> None:
    """Soft-delete one Event memory while retaining its audit history."""

    source = await _load_mutable_event_memory(
        session,
        organization_id=organization_id,
        user_id=user_id,
        event_id=event.id,
        memory_id=memory_id,
        allow_deleted=True,
    )
    if source.status == MemoryStatus.DELETED.value:
        return
    if source.status != MemoryStatus.ACTIVE.value:
        raise EventMemoryMutationError("Event memory is no longer active")
    record = row_to_record(source)
    catalog = MemoryCatalog([record])
    try:
        catalog.delete(record.id, record.scope)
    except ValueError as exc:
        raise EventMemoryMutationError(str(exc)) from exc
    record_to_row(catalog.records()[0], source)
    now = datetime.now(timezone.utc)
    session.add(source)
    session.add(
        EventTrailEntry(
            organization_id=organization_id,
            user_id=user_id,
            event_id=event.id,
            actor_id=user_id,
            kind="event_memory_forgotten",
            summary="Forgot an Event memory",
            evidence_refs=[{"knowledge_entry_id": source.id}],
            payload={"kind": source.kind, "sensitivity": source.sensitivity},
        )
    )
    event.updated_at = now
    session.add(event)
    await session.flush()
    await _refresh_event_context(session, event=event)


async def promote_event_memory(
    session: AsyncSession,
    *,
    event: Event,
    organization_id: str,
    user_id: str,
    memory_id: str,
) -> tuple[KnowledgeEntry, bool]:
    """Copy one Event memory into global user memory, idempotently."""

    source = await _load_mutable_event_memory(
        session,
        organization_id=organization_id,
        user_id=user_id,
        event_id=event.id,
        memory_id=memory_id,
    )
    promoted_id = _promotion_id(organization_id, user_id, source.id)
    existing = await session.get(KnowledgeEntry, promoted_id)
    if existing is not None:
        if (
            existing.organization_id != organization_id
            or existing.user_id != user_id
            or existing.event_id is not None
        ):
            raise EventMemoryMutationError("Global memory promotion is unavailable")
        if existing.status == MemoryStatus.ACTIVE.value:
            return existing, False

    promoted = create_record(
        user_id=user_id,
        organization_id=organization_id,
        content=source.content,
        tags=list(source.tags or []),
        importance=source.importance,
        kind=source.kind,
        confidence=source.confidence,
        sensitivity=source.sensitivity,
        source="user_promotion",
        source_id=source.id,
        retention=dict(source.retention or {}),
        conflict_key=source.conflict_key,
        event_at=source.event_at,
        metadata={
            "promoted_from_event_id": event.id,
            "promoted_from_memory_id": source.id,
        },
    )
    promoted.id = promoted_id
    if existing is not None:
        promoted.created_at = existing.created_at

    if promoted.conflict_key:
        global_result = await session.execute(
            select(KnowledgeEntry).where(
                KnowledgeEntry.organization_id == organization_id,
                KnowledgeEntry.user_id == user_id,
                col(KnowledgeEntry.event_id).is_(None),
                KnowledgeEntry.status == MemoryStatus.ACTIVE.value,
                KnowledgeEntry.id != promoted_id,
            )
        )
        global_rows = list(global_result.scalars().all())
        catalog = MemoryCatalog(row_to_record(row) for row in global_rows)
        catalog.add(promoted, conflict_policy=ConflictPolicy.SUPERSEDE)
        rows_by_id = {row.id: row for row in global_rows}
        for record in catalog.records():
            if record.id in rows_by_id:
                record_to_row(record, rows_by_id[record.id])
                session.add(rows_by_id[record.id])

    promoted_row = record_to_row(promoted, existing)
    source_metadata = dict(source.metadata_ or {})
    source_metadata["promoted_global_id"] = promoted_id
    source_metadata["promoted_at"] = datetime.now(timezone.utc).isoformat()
    source.metadata_ = source_metadata
    session.add(source)
    session.add(promoted_row)
    session.add(
        EventTrailEntry(
            organization_id=organization_id,
            user_id=user_id,
            event_id=event.id,
            actor_id=user_id,
            kind="event_memory_promoted",
            summary="Made an Event memory available across Aloy",
            evidence_refs=[
                {"knowledge_entry_id": source.id},
                {"global_knowledge_entry_id": promoted_id},
            ],
            payload={"kind": source.kind, "sensitivity": source.sensitivity},
        )
    )
    event.updated_at = datetime.now(timezone.utc)
    session.add(event)
    await session.flush()
    return promoted_row, True


async def _load_mutable_event_memory(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    event_id: str,
    memory_id: str,
    allow_deleted: bool = False,
) -> KnowledgeEntry:
    row = await session.get(KnowledgeEntry, memory_id)
    if (
        row is None
        or row.organization_id != organization_id
        or row.user_id != user_id
        or row.event_id != event_id
        or row.scope_level == ORG
    ):
        raise EventMemoryMutationError("Event memory not found", status_code=404)
    if not allow_deleted and row.status != MemoryStatus.ACTIVE.value:
        raise EventMemoryMutationError("Event memory is no longer active")
    if not _is_user_managed_memory(row):
        raise EventMemoryMutationError(
            "Canonical Event evidence and records are read-only here"
        )
    return row


async def _refresh_event_context(session: AsyncSession, *, event: Event) -> None:
    active_briefs = (
        (
            await session.execute(
                select(EventBrief).where(
                    EventBrief.organization_id == event.organization_id,
                    EventBrief.user_id == event.user_id,
                    EventBrief.event_id == event.id,
                    EventBrief.status == "active",
                )
            )
        )
        .scalars()
        .all()
    )
    now = datetime.now(timezone.utc)
    for brief in active_briefs:
        brief.status = "stale"
        brief.updated_at = now
        session.add(brief)
    await session.flush()
    await refresh_event_context_snapshot(
        session,
        organization_id=event.organization_id,
        user_id=event.user_id,
        event_id=event.id,
    )
    await queue_event_bootstrap_if_ready(
        session,
        organization_id=event.organization_id,
        user_id=event.user_id,
        event_id=event.id,
    )


def _promotion_id(organization_id: str, user_id: str, memory_id: str) -> str:
    digest = hashlib.sha256(
        f"{organization_id}:{user_id}:{memory_id}".encode("utf-8")
    ).hexdigest()[:32]
    return f"memory_global_{digest}"
