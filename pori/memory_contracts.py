"""Provider-agnostic long-term memory contracts: ``MemoryRecord`` (the
kind/scope/provenance/retention/sensitivity envelope), the
``MemoryRepository`` protocol backends implement, and ``MemoryCatalog``, which
enforces scope filtering and conflict policy on top of any repository.
Invariant: all timestamps are timezone-aware UTC (``utc_now`` / ``as_utc``).
Consumed by product backends (e.g. the Aloy backend's knowledge store) rather
than the agent loop itself.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class MemoryKind(str, Enum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class MemorySensitivity(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class ConflictPolicy(str, Enum):
    KEEP_BOTH = "keep_both"
    REJECT = "reject"
    SUPERSEDE = "supersede"


class MemoryScope(BaseModel):
    organization_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    agent_id: Optional[str] = None
    session_id: Optional[str] = None

    @property
    def namespace(self) -> str:
        return ":".join(
            [
                self.organization_id,
                self.user_id,
                self.agent_id or "*",
                self.session_id or "*",
            ]
        )

    def can_access(self, record_scope: "MemoryScope") -> bool:
        """Return whether this request scope may read the record scope."""
        if self.organization_id != record_scope.organization_id:
            return False
        if self.user_id != record_scope.user_id:
            return False
        if record_scope.agent_id not in (None, self.agent_id):
            return False
        if record_scope.session_id not in (None, self.session_id):
            return False
        return True


class MemoryProvenance(BaseModel):
    source: str = Field(default="agent", min_length=1)
    source_id: Optional[str] = None
    actor_id: Optional[str] = None
    conversation_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryRetention(BaseModel):
    ttl_days: Optional[int] = Field(default=None, ge=1)
    delete_after: Optional[datetime] = None
    legal_hold: bool = False

    def expires_at(self, created_at: datetime) -> Optional[datetime]:
        explicit = as_utc(self.delete_after) if self.delete_after else None
        ttl_expiry = (
            as_utc(created_at) + timedelta(days=self.ttl_days)
            if self.ttl_days is not None
            else None
        )
        if explicit is None:
            return ttl_expiry
        if ttl_expiry is None:
            return explicit
        return min(explicit, ttl_expiry)


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:16]}")
    scope: MemoryScope
    kind: MemoryKind = MemoryKind.SEMANTIC
    content: str = Field(min_length=1)
    tags: List[str] = Field(default_factory=list)
    importance: int = Field(default=1, ge=1, le=5)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    sensitivity: MemorySensitivity = MemorySensitivity.INTERNAL
    provenance: MemoryProvenance = Field(default_factory=MemoryProvenance)
    retention: MemoryRetention = Field(default_factory=MemoryRetention)
    conflict_key: Optional[str] = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    superseded_by: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    event_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.retention.legal_hold:
            return False
        expiry = self.retention.expires_at(self.created_at)
        return expiry is not None and expiry <= as_utc(now or utc_now())

    def is_retrievable(self, now: Optional[datetime] = None) -> bool:
        return self.status == MemoryStatus.ACTIVE and not self.is_expired(now)


class MemoryHit(BaseModel):
    record: MemoryRecord
    semantic_score: float = Field(default=0.0, ge=0.0, le=1.0)
    lexical_score: float = Field(default=0.0, ge=0.0, le=1.0)
    final_score: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_by: List[str] = Field(default_factory=list)


class RetrievalEvaluation(BaseModel):
    recall_at_k: float = Field(ge=0.0, le=1.0)
    precision_at_k: float = Field(ge=0.0, le=1.0)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    leaked_record_ids: List[str] = Field(default_factory=list)


class MemoryRepository(Protocol):
    def list_records(self) -> List[MemoryRecord]: ...

    def replace_records(self, records: List[MemoryRecord]) -> None: ...


class MemoryCatalog:
    """Storage-neutral memory policy, mutation, retrieval, and export boundary."""

    def __init__(self, records: Optional[Iterable[MemoryRecord]] = None):
        self._records = list(records or [])

    def records(self) -> List[MemoryRecord]:
        return list(self._records)

    def add(
        self,
        record: MemoryRecord,
        conflict_policy: ConflictPolicy = ConflictPolicy.KEEP_BOTH,
    ) -> MemoryRecord:
        conflicts = [
            existing
            for existing in self._records
            if existing.status == MemoryStatus.ACTIVE
            and record.conflict_key
            and existing.conflict_key == record.conflict_key
            and existing.scope == record.scope
            and existing.kind == record.kind
        ]
        if conflicts and conflict_policy == ConflictPolicy.REJECT:
            raise ValueError(
                f"Active memory already exists for conflict key '{record.conflict_key}'"
            )
        if conflicts and conflict_policy == ConflictPolicy.SUPERSEDE:
            now = utc_now()
            for existing in conflicts:
                existing.status = MemoryStatus.SUPERSEDED
                existing.superseded_by = record.id
                existing.updated_at = now
        self._records.append(record)
        return record

    def delete(
        self,
        record_id: str,
        scope: MemoryScope,
        *,
        hard: bool = False,
    ) -> bool:
        for index, record in enumerate(self._records):
            if record.id != record_id or not scope.can_access(record.scope):
                continue
            if record.retention.legal_hold:
                raise ValueError("Memory is under legal hold")
            if hard:
                self._records.pop(index)
            else:
                record.status = MemoryStatus.DELETED
                record.deleted_at = utc_now()
                record.updated_at = record.deleted_at
            return True
        return False

    def prune_expired(self, now: Optional[datetime] = None) -> List[str]:
        current = now or utc_now()
        expired = [
            record.id
            for record in self._records
            if record.is_expired(current) and not record.retention.legal_hold
        ]
        self._records = [record for record in self._records if record.id not in expired]
        return expired

    def export(
        self,
        scope: MemoryScope,
        *,
        include_deleted: bool = False,
    ) -> List[MemoryRecord]:
        return [
            record
            for record in self._records
            if scope.can_access(record.scope)
            and (include_deleted or record.status != MemoryStatus.DELETED)
        ]

    def search(
        self,
        scope: MemoryScope,
        scorer: Callable[[MemoryRecord], MemoryHit],
        *,
        k: int = 5,
        kinds: Optional[Iterable[MemoryKind]] = None,
        tags: Optional[Iterable[str]] = None,
        min_score: float = 0.0,
        max_sensitivity: MemorySensitivity = MemorySensitivity.RESTRICTED,
    ) -> List[MemoryHit]:
        kind_filter = set(kinds or [])
        tag_filter = {str(tag) for tag in (tags or [])}
        sensitivity_order = list(MemorySensitivity)
        max_sensitivity_index = sensitivity_order.index(max_sensitivity)
        hits: List[MemoryHit] = []
        for record in self._records:
            if not scope.can_access(record.scope) or not record.is_retrievable():
                continue
            if kind_filter and record.kind not in kind_filter:
                continue
            if tag_filter and not tag_filter.intersection(record.tags):
                continue
            if sensitivity_order.index(record.sensitivity) > max_sensitivity_index:
                continue
            hit = scorer(record)
            if hit.final_score >= min_score:
                hits.append(hit)
        hits.sort(
            key=lambda hit: (hit.final_score, hit.record.updated_at),
            reverse=True,
        )
        return hits[: max(0, k)]


def evaluate_retrieval(
    hits: Iterable[MemoryHit],
    expected_record_ids: Iterable[str],
    request_scope: MemoryScope,
) -> RetrievalEvaluation:
    hit_list = list(hits)
    expected = set(expected_record_ids)
    returned = [hit.record.id for hit in hit_list]
    relevant = [record_id for record_id in returned if record_id in expected]
    first_rank = next(
        (
            index
            for index, record_id in enumerate(returned, start=1)
            if record_id in expected
        ),
        None,
    )
    leaked = [
        hit.record.id
        for hit in hit_list
        if not request_scope.can_access(hit.record.scope)
    ]
    return RetrievalEvaluation(
        recall_at_k=(len(set(relevant)) / len(expected)) if expected else 1.0,
        precision_at_k=(len(relevant) / len(returned)) if returned else 0.0,
        reciprocal_rank=(1.0 / first_rank) if first_rank else 0.0,
        leaked_record_ids=leaked,
    )


__all__ = [
    "ConflictPolicy",
    "MemoryCatalog",
    "MemoryHit",
    "MemoryKind",
    "MemoryProvenance",
    "MemoryRecord",
    "MemoryRepository",
    "MemoryRetention",
    "MemoryScope",
    "MemorySensitivity",
    "MemoryStatus",
    "RetrievalEvaluation",
    "as_utc",
    "evaluate_retrieval",
    "utc_now",
]
