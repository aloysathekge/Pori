"""Bridges the kernel's ``MemoryRecord`` contract onto the backend's
``KnowledgeEntry`` rows: row↔record converters, request-scope construction
(per-user tenant boundary until organization membership owns memory),
conflict-policy application, and search. Used by the memory routes and by run
surfaces that read/write long-term knowledge.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence

from pori import (
    ConflictPolicy,
    MemoryCatalog,
    MemoryHit,
    MemoryKind,
    MemoryProvenance,
    MemoryRecord,
    MemoryRetention,
    MemoryScope,
    MemorySensitivity,
)

from .models import KnowledgeEntry
from .schemas import KnowledgeEntryResponse


# Product features may have historically stored their provenance category in
# ``KnowledgeEntry.kind``. The kernel contract intentionally keeps this field
# limited to cognitive memory kinds. Translate known legacy values at the
# backend boundary while preserving the original category in record metadata.
_LEGACY_MEMORY_KIND_MAP: dict[str, MemoryKind] = {
    "template_context": MemoryKind.SEMANTIC,
}


def _canonical_memory_kind(kind: str) -> MemoryKind:
    legacy_kind = _LEGACY_MEMORY_KIND_MAP.get(kind)
    if legacy_kind is not None:
        return legacy_kind
    return MemoryKind(kind)


def personal_organization_id(user_id: str) -> str:
    """Temporary tenant boundary until organization membership exists."""
    return f"user:{user_id}"


def request_scope(
    user_id: str,
    *,
    organization_id: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    event_id: str | None = None,
) -> MemoryScope:
    return MemoryScope(
        organization_id=organization_id or personal_organization_id(user_id),
        user_id=user_id,
        event_id=event_id,
        agent_id=agent_id,
        session_id=session_id,
    )


def row_to_record(row: KnowledgeEntry) -> MemoryRecord:
    kind = _canonical_memory_kind(row.kind)
    metadata = dict(row.metadata_ or {})
    if kind.value != row.kind:
        metadata.setdefault("source_memory_kind", row.kind)
    provenance = row.provenance or {
        "source": row.source,
        "metadata": metadata,
    }
    return MemoryRecord.model_validate(
        {
            "id": row.id,
            "scope": {
                "organization_id": row.organization_id,
                "user_id": row.user_id,
                "event_id": row.event_id,
                "agent_id": row.agent_id,
                "session_id": row.session_id,
            },
            "kind": kind,
            "content": row.content,
            "tags": row.tags or [],
            "importance": row.importance,
            "confidence": row.confidence,
            "sensitivity": row.sensitivity,
            "provenance": provenance,
            "retention": row.retention or {},
            "conflict_key": row.conflict_key,
            "status": row.status,
            "superseded_by": row.superseded_by,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "event_at": row.event_at,
            "deleted_at": row.deleted_at,
            "metadata": metadata,
        }
    )


def record_to_row(
    record: MemoryRecord, row: KnowledgeEntry | None = None
) -> KnowledgeEntry:
    values = {
        "id": record.id,
        "organization_id": record.scope.organization_id,
        "user_id": record.scope.user_id,
        "event_id": record.scope.event_id,
        "agent_id": record.scope.agent_id,
        "session_id": record.scope.session_id,
        "content": record.content,
        "tags": record.tags,
        "importance": record.importance,
        "kind": record.kind.value,
        "confidence": record.confidence,
        "sensitivity": record.sensitivity.value,
        "source": record.provenance.source,
        "provenance": record.provenance.model_dump(mode="json"),
        "retention": record.retention.model_dump(mode="json"),
        "conflict_key": record.conflict_key,
        "status": record.status.value,
        "superseded_by": record.superseded_by,
        "metadata_": record.metadata,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "event_at": record.event_at,
        "deleted_at": record.deleted_at,
    }
    if row is None:
        return KnowledgeEntry(**values)
    for name, value in values.items():
        setattr(row, name, value)
    return row


def record_response(record: MemoryRecord) -> KnowledgeEntryResponse:
    return KnowledgeEntryResponse(
        id=record.id,
        organization_id=record.scope.organization_id,
        user_id=record.scope.user_id,
        event_id=record.scope.event_id,
        agent_id=record.scope.agent_id,
        session_id=record.scope.session_id,
        content=record.content,
        tags=record.tags,
        importance=record.importance,
        kind=record.kind.value,
        confidence=record.confidence,
        sensitivity=record.sensitivity.value,
        source=record.provenance.source,
        provenance=record.provenance.model_dump(mode="json"),
        retention=record.retention.model_dump(mode="json"),
        conflict_key=record.conflict_key,
        status=record.status.value,
        superseded_by=record.superseded_by,
        created_at=record.created_at.isoformat(),
        updated_at=record.updated_at.isoformat(),
        event_at=record.event_at.isoformat() if record.event_at else None,
    )


def create_record(
    *,
    user_id: str,
    organization_id: str | None = None,
    content: str,
    agent_id: str | None = None,
    session_id: str | None = None,
    event_id: str | None = None,
    tags: list[str] | None = None,
    importance: int = 1,
    kind: str = "semantic",
    confidence: float = 1.0,
    sensitivity: str = "internal",
    source: str = "user",
    source_id: str | None = None,
    conversation_id: str | None = None,
    run_id: str | None = None,
    retention: dict | None = None,
    conflict_key: str | None = None,
    event_at: datetime | None = None,
    metadata: dict | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        scope=request_scope(
            user_id,
            organization_id=organization_id,
            event_id=event_id,
            agent_id=agent_id,
            session_id=session_id,
        ),
        content=content,
        tags=tags or [],
        importance=importance,
        kind=MemoryKind(kind),
        confidence=confidence,
        sensitivity=MemorySensitivity(sensitivity),
        provenance=MemoryProvenance(
            source=source,
            source_id=source_id,
            actor_id=user_id,
            conversation_id=conversation_id,
            run_id=run_id,
        ),
        retention=MemoryRetention.model_validate(retention or {}),
        conflict_key=conflict_key,
        event_at=event_at,
        metadata=metadata or {},
    )


def apply_conflict_policy(
    records: Iterable[MemoryRecord],
    record: MemoryRecord,
    policy: str,
) -> MemoryCatalog:
    catalog = MemoryCatalog(records)
    catalog.add(record, conflict_policy=ConflictPolicy(policy))
    return catalog


def search_records(
    records: Iterable[MemoryRecord],
    *,
    scope: MemoryScope,
    query: str,
    k: int,
    kinds: Sequence[str] | None,
    tags: list[str] | None,
    min_score: float,
) -> list[MemoryHit]:
    terms = {term for term in query.lower().split() if term}

    def scorer(record: MemoryRecord) -> MemoryHit:
        content_terms = {term for term in record.content.lower().split() if term}
        overlap = len(terms.intersection(content_terms))
        lexical = overlap / max(1, len(terms.union(content_terms)))
        exact = 1.0 if query.lower() in record.content.lower() else 0.0
        final = min(
            1.0,
            (0.65 * lexical)
            + (0.25 * exact)
            + (0.05 * record.confidence)
            + (0.01 * record.importance),
        )
        return MemoryHit(
            record=record,
            lexical_score=lexical,
            final_score=final,
            matched_by=["lexical"] if lexical or exact else [],
        )

    return MemoryCatalog(records).search(
        scope,
        scorer,
        k=k,
        kinds=[MemoryKind(kind) for kind in (kinds or [])],
        tags=tags,
        min_score=min_score,
    )
