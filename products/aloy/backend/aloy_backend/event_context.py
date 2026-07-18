"""Versioned, evidence-grounded context assembly for one Aloy Event."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from pori import stable_fingerprint

from .models import (
    ActionProposal,
    Event,
    EventBrief,
    EventConnectionGrant,
    EventContextSnapshot,
    EventSetupContextItem,
    EventTrailEntry,
    KnowledgeEntry,
    StoredFile,
    Task,
)
from .surface_state import surface_state_context_projection

ReadinessLevel = Literal["not_applicable", "name_only", "little", "sufficient", "rich"]
EvidenceKind = Literal[
    "event",
    "knowledge_entry",
    "context_item",
    "task",
    "proposal",
    "file",
    "trail",
    "surface_entity",
]
MAX_BOOTSTRAP_EVIDENCE_CHARS = 100_000
MAX_BOOTSTRAP_EVIDENCE_ITEM_CHARS = 16_000
_BOOTSTRAP_LIFECYCLE_TRAIL_KINDS = {
    "event_bootstrap_queued",
    "event_bootstrap_retried",
    "event_bootstrap_started",
    "event_bootstrap_superseded",
    "event_bootstrap_retry_scheduled",
    "event_bootstrap_failed",
    "event_brief_published",
}


class EventEvidenceRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: EvidenceKind
    id: str = Field(min_length=1)


class GroundedText(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=4_000)
    evidence_refs: list[EventEvidenceRef] = Field(min_length=1, max_length=30)


class EventBriefPayload(BaseModel):
    """Structured output contract for the later Event-bootstrap model Run."""

    model_config = ConfigDict(str_strip_whitespace=True)

    purpose: GroundedText
    outcomes: list[GroundedText] = Field(default_factory=list, max_length=20)
    time_horizon: GroundedText | None = None
    important_entities: list[GroundedText] = Field(default_factory=list, max_length=50)
    important_dates: list[GroundedText] = Field(default_factory=list, max_length=50)
    recurring_work: list[GroundedText] = Field(default_factory=list, max_length=30)
    available_sources: list[GroundedText] = Field(default_factory=list, max_length=30)
    unknowns: list[str] = Field(default_factory=list, max_length=50)
    user_jobs: list[GroundedText] = Field(default_factory=list, max_length=30)


class ContextReadiness(BaseModel):
    model_config = ConfigDict(frozen=True)

    level: ReadinessLevel
    bootstrap_eligible: bool
    should_bootstrap: bool
    evidence_chars: int
    ready_source_count: int
    available_connection_count: int
    reasons: list[str]


class EventContextPack(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "1"
    event: dict[str, Any]
    canonical_state: dict[str, Any]
    readiness: ContextReadiness
    active_brief: dict[str, Any] | None = None
    evidence_catalog: list[dict[str, Any]] = Field(default_factory=list)
    cache_policy: dict[str, Any]


def _utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _bounded_json(value: Any, *, max_chars: int = 4_000) -> Any:
    """Keep the stable prompt pack bounded while preserving change identity."""
    if value is None:
        return None
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(serialized) <= max_chars:
        return value
    return {
        "truncated": True,
        "fingerprint": stable_fingerprint(value),
        "preview": serialized[:max_chars],
    }


def _snapshot_evidence(knowledge: list[KnowledgeEntry]) -> list[dict[str, Any]]:
    """Freeze a bounded evidence body set for the later bootstrap model Run."""
    remaining = MAX_BOOTSTRAP_EVIDENCE_CHARS
    payload: list[dict[str, Any]] = []
    for entry in knowledge:
        if remaining <= 0:
            break
        full = entry.content.strip()
        if not full:
            continue
        limit = min(MAX_BOOTSTRAP_EVIDENCE_ITEM_CHARS, remaining)
        text = full[:limit]
        payload.append(
            {
                "kind": "knowledge_entry",
                "id": entry.id,
                "text": text,
                "truncated": len(text) < len(full),
                "content_fingerprint": stable_fingerprint(full),
                "tags": entry.tags or [],
                "sensitivity": entry.sensitivity,
                "provenance": _bounded_json(entry.provenance, max_chars=2_000),
            }
        )
        remaining -= len(text)
    return payload


def _readiness(
    event: Event,
    knowledge: list[KnowledgeEntry],
    context_items: list[EventSetupContextItem],
    connections: list[EventConnectionGrant],
    *,
    has_brief: bool,
) -> ContextReadiness:
    if event.is_life:
        return ContextReadiness(
            level="not_applicable",
            bootstrap_eligible=False,
            should_bootstrap=False,
            evidence_chars=0,
            ready_source_count=0,
            available_connection_count=len(connections),
            reasons=[
                "Life uses Conversations and canonical state without a bootstrap Surface."
            ],
        )
    active_knowledge = [entry for entry in knowledge if entry.status == "active"]
    evidence_chars = sum(len(entry.content.strip()) for entry in active_knowledge)
    if not evidence_chars:
        evidence_chars = len(event.summary.strip())
    ready_sources = [
        item
        for item in context_items
        if item.kind in {"file", "link"} and item.status == "ready"
    ]
    reasons: list[str] = []
    if not event.summary.strip() and not active_knowledge and not ready_sources:
        level: ReadinessLevel = "name_only"
        reasons.append("Only the Event name is known.")
    elif evidence_chars >= 4_000 and len(ready_sources) >= 2:
        level = "rich"
        reasons.append("Several ready sources provide substantial accepted evidence.")
    elif evidence_chars >= 500 or (ready_sources and evidence_chars >= 200):
        level = "sufficient"
        reasons.append("Accepted evidence is sufficient for a grounded first brief.")
    else:
        level = "little"
        reasons.append(
            "Some context exists, but important structure is still uncertain."
        )
    if any(item.status in {"pending", "ingesting"} for item in context_items):
        reasons.append("Additional setup sources are still processing.")
    if any(item.status == "failed" for item in context_items):
        reasons.append("At least one setup source could not be ingested.")
    return ContextReadiness(
        level=level,
        bootstrap_eligible=True,
        should_bootstrap=level in {"sufficient", "rich"} and not has_brief,
        evidence_chars=evidence_chars,
        ready_source_count=len(ready_sources),
        available_connection_count=len(connections),
        reasons=reasons,
    )


async def refresh_event_context_snapshot(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    event_id: str,
) -> tuple[EventContextSnapshot, EventContextPack, bool]:
    """Build canonical context and append a snapshot only when content changes."""
    event = await session.get(Event, event_id)
    if (
        event is None
        or event.organization_id != organization_id
        or event.user_id != user_id
    ):
        raise ValueError("Event is unavailable")

    async def rows(model, *filters, order=None, limit=100):
        statement = select(model).where(*filters)
        if order is not None:
            statement = statement.order_by(order)
        return list((await session.execute(statement.limit(limit))).scalars().all())

    owner = (organization_id, user_id, event.id)
    tasks = await rows(
        Task,
        Task.organization_id == owner[0],
        Task.user_id == owner[1],
        Task.event_id == owner[2],
        order=col(Task.order),
        limit=100,
    )
    proposals = await rows(
        ActionProposal,
        ActionProposal.organization_id == owner[0],
        ActionProposal.user_id == owner[1],
        ActionProposal.event_id == owner[2],
        order=col(ActionProposal.updated_at).desc(),
        limit=50,
    )
    files = await rows(
        StoredFile,
        StoredFile.organization_id == owner[0],
        StoredFile.user_id == owner[1],
        StoredFile.event_id == owner[2],
        order=col(StoredFile.created_at).desc(),
        limit=100,
    )
    trails = await rows(
        EventTrailEntry,
        EventTrailEntry.organization_id == owner[0],
        EventTrailEntry.user_id == owner[1],
        EventTrailEntry.event_id == owner[2],
        ~col(EventTrailEntry.kind).in_(_BOOTSTRAP_LIFECYCLE_TRAIL_KINDS),
        order=col(EventTrailEntry.created_at).desc(),
        limit=30,
    )
    context_items = await rows(
        EventSetupContextItem,
        EventSetupContextItem.organization_id == owner[0],
        EventSetupContextItem.user_id == owner[1],
        EventSetupContextItem.event_id == owner[2],
        order=col(EventSetupContextItem.created_at),
        limit=100,
    )
    connections = await rows(
        EventConnectionGrant,
        EventConnectionGrant.organization_id == owner[0],
        EventConnectionGrant.user_id == owner[1],
        EventConnectionGrant.event_id == owner[2],
        limit=50,
    )
    knowledge = await rows(
        KnowledgeEntry,
        KnowledgeEntry.organization_id == owner[0],
        KnowledgeEntry.user_id == owner[1],
        KnowledgeEntry.event_id == owner[2],
        KnowledgeEntry.status == "active",
        order=col(KnowledgeEntry.updated_at).desc(),
        limit=100,
    )
    active_brief = (
        (
            await session.execute(
                select(EventBrief)
                .where(
                    EventBrief.organization_id == owner[0],
                    EventBrief.user_id == owner[1],
                    EventBrief.event_id == owner[2],
                    EventBrief.status == "active",
                )
                .order_by(col(EventBrief.version).desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    surface_state = await surface_state_context_projection(
        session,
        organization_id=owner[0],
        user_id=owner[1],
        event_id=owner[2],
    )

    readiness = _readiness(
        event, knowledge, context_items, connections, has_brief=active_brief is not None
    )
    evidence_payload = _snapshot_evidence(knowledge)
    evidence_refs = (
        [{"kind": "event", "id": event.id}]
        + [{"kind": "knowledge_entry", "id": item["id"]} for item in evidence_payload]
        + [
            {"kind": "context_item", "id": item.id}
            for item in context_items
            if item.status == "ready"
        ]
    )
    evidence_refs += [{"kind": "task", "id": item.id} for item in tasks]
    evidence_refs += [{"kind": "proposal", "id": item.id} for item in proposals]
    evidence_refs += [{"kind": "file", "id": item.id} for item in files]
    evidence_refs += [{"kind": "trail", "id": item.id} for item in trails]
    evidence_refs += [
        {
            "kind": "surface_entity",
            "id": f"{item['namespace']}:{item['key']}",
        }
        for item in (surface_state or {}).get("records", [])
    ]

    sensitive = any(
        entry.sensitivity in {"confidential", "restricted"} for entry in knowledge
    ) or any(
        item.sensitivity in {"confidential", "restricted"} for item in context_items
    )
    canonical = {
        "tasks": [
            {
                "id": item.id,
                "title": item.title,
                "status": item.status,
                "priority": item.priority,
                "due_at": _utc(item.due_at),
                "blocker": item.blocker,
                "result_summary": item.result_summary,
            }
            for item in tasks
        ],
        "proposals": [
            {
                "id": item.id,
                "tool": item.tool,
                "reason": item.reason,
                "impact": item.impact,
                "risk": item.risk,
                "status": item.status,
                "receipt": _bounded_json(item.receipt),
            }
            for item in proposals
        ],
        "files": [
            {"id": item.id, "name": item.name, "kind": item.kind} for item in files
        ],
        "connections": [
            {
                "provider": item.provider,
                "status": item.status,
                "access_scope": item.access_scope,
            }
            for item in connections
        ],
        "recent_trail": [
            {
                "id": item.id,
                "kind": item.kind,
                "summary": item.summary,
                "created_at": _utc(item.created_at),
            }
            for item in trails
        ],
        "surface_state": surface_state,
    }
    catalog = [
        {
            "kind": "knowledge_entry",
            "id": item.id,
            "tags": item.tags or [],
            "sensitivity": item.sensitivity,
            "content_chars": len(item.content),
            "content_fingerprint": stable_fingerprint(item.content),
            "updated_at": _utc(item.updated_at),
        }
        for item in knowledge
    ] + [
        {
            "kind": "context_item",
            "id": item.id,
            "source_kind": item.kind,
            "label": item.label,
            "status": item.status,
            "sensitivity": item.sensitivity,
            "source_fingerprint": item.sha256 or None,
            "retrieved_at": _utc(item.retrieved_at),
        }
        for item in context_items
    ]
    pack = EventContextPack(
        event={
            "id": event.id,
            "title": event.title,
            "type": event.type,
            "lifecycle": event.lifecycle,
            "phase": event.phase,
            "summary": event.summary,
            "is_life": event.is_life,
        },
        canonical_state=canonical,
        readiness=readiness,
        active_brief=(
            {
                "id": active_brief.id,
                "version": active_brief.version,
                "payload": active_brief.payload,
                "evidence_refs": active_brief.evidence_refs,
            }
            if active_brief
            else None
        ),
        evidence_catalog=catalog,
        cache_policy={
            "provider_cache_allowed": not sensitive,
            "reason": "sensitive_evidence" if sensitive else "stable_event_prefix",
        },
    )
    serialized = pack.model_dump(mode="json")
    fingerprint = stable_fingerprint(serialized)
    existing = (
        (
            await session.execute(
                select(EventContextSnapshot).where(
                    EventContextSnapshot.event_id == event.id,
                    EventContextSnapshot.fingerprint == fingerprint,
                )
            )
        )
        .scalars()
        .first()
    )
    if existing is not None:
        return existing, pack, False
    next_version = (
        int(
            (
                await session.execute(
                    select(
                        func.coalesce(func.max(EventContextSnapshot.version), 0)
                    ).where(EventContextSnapshot.event_id == event.id)
                )
            ).scalar_one()
        )
        + 1
    )
    snapshot = EventContextSnapshot(
        organization_id=organization_id,
        user_id=user_id,
        event_id=event.id,
        version=next_version,
        fingerprint=fingerprint,
        readiness=readiness.level,
        provider_cache_allowed=not sensitive,
        pack=serialized,
        evidence_refs=evidence_refs,
        evidence_payload=evidence_payload,
    )
    session.add(snapshot)
    await session.flush()
    return snapshot, pack, True


def render_event_context_pack(snapshot: EventContextSnapshot) -> str:
    """Stable prompt representation; evidence bodies remain in retrieval."""
    return json.dumps(
        {
            "snapshot_id": snapshot.id,
            "snapshot_version": snapshot.version,
            **snapshot.pack,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def render_event_bootstrap_input(snapshot: EventContextSnapshot) -> str:
    """Render the exact frozen pack plus private bounded evidence for bootstrap."""
    return json.dumps(
        {
            "snapshot_id": snapshot.id,
            "snapshot_version": snapshot.version,
            "context": snapshot.pack,
            "evidence": snapshot.evidence_payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _brief_refs(payload: EventBriefPayload) -> list[EventEvidenceRef]:
    grounded: list[GroundedText] = [payload.purpose]
    grounded.extend(payload.outcomes)
    grounded.extend(payload.important_entities)
    grounded.extend(payload.important_dates)
    grounded.extend(payload.recurring_work)
    grounded.extend(payload.available_sources)
    grounded.extend(payload.user_jobs)
    if payload.time_horizon:
        grounded.append(payload.time_horizon)
    unique: dict[tuple[str, str], EventEvidenceRef] = {}
    for item in grounded:
        for ref in item.evidence_refs:
            unique[(ref.kind, ref.id)] = ref
    return list(unique.values())


async def publish_event_brief(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    event_id: str,
    snapshot_id: str,
    payload: EventBriefPayload,
    creator_run_id: str | None = None,
) -> tuple[EventBrief, bool]:
    """Validate evidence and atomically make one idempotent brief active."""
    snapshot = await session.get(EventContextSnapshot, snapshot_id)
    if (
        snapshot is None
        or snapshot.event_id != event_id
        or snapshot.organization_id != organization_id
        or snapshot.user_id != user_id
    ):
        raise ValueError("Event context snapshot is unavailable")
    if snapshot.readiness not in {"sufficient", "rich"}:
        raise ValueError("Event context is not ready for a grounded brief")
    refs = _brief_refs(payload)
    allowed = {
        (str(item.get("kind")), str(item.get("id"))) for item in snapshot.evidence_refs
    }
    missing = sorted(
        f"{ref.kind}:{ref.id}" for ref in refs if (ref.kind, ref.id) not in allowed
    )
    if missing:
        raise ValueError(
            "Brief cites evidence outside the Event snapshot: " + ", ".join(missing)
        )
    serialized = payload.model_dump(mode="json")
    fingerprint = stable_fingerprint(
        {"snapshot_id": snapshot.id, "payload": serialized}
    )
    existing = (
        (
            await session.execute(
                select(EventBrief).where(
                    EventBrief.event_id == event_id,
                    EventBrief.fingerprint == fingerprint,
                )
            )
        )
        .scalars()
        .first()
    )
    if existing is not None:
        return existing, False
    now = datetime.now(timezone.utc)
    active = list(
        (
            await session.execute(
                select(EventBrief).where(
                    EventBrief.event_id == event_id,
                    EventBrief.organization_id == organization_id,
                    EventBrief.user_id == user_id,
                    EventBrief.status == "active",
                )
            )
        )
        .scalars()
        .all()
    )
    for item in active:
        item.status = "superseded"
        item.updated_at = now
        session.add(item)
    next_version = (
        int(
            (
                await session.execute(
                    select(func.coalesce(func.max(EventBrief.version), 0)).where(
                        EventBrief.event_id == event_id
                    )
                )
            ).scalar_one()
        )
        + 1
    )
    brief = EventBrief(
        organization_id=organization_id,
        user_id=user_id,
        event_id=event_id,
        version=next_version,
        source_context_snapshot_id=snapshot.id,
        creator_run_id=creator_run_id,
        fingerprint=fingerprint,
        payload=serialized,
        evidence_refs=[ref.model_dump(mode="json") for ref in refs],
    )
    session.add(brief)
    await session.flush()
    session.add(
        EventTrailEntry(
            organization_id=organization_id,
            user_id=user_id,
            event_id=event_id,
            actor_id="aloy:event-bootstrap",
            kind="event_brief_published",
            summary=f"Published Event Brief version {next_version}",
            run_id=creator_run_id,
            evidence_refs=[
                {"event_brief_id": brief.id},
                {"context_snapshot_id": snapshot.id},
                *[ref.model_dump(mode="json") for ref in refs],
            ],
            payload={"version": next_version, "fingerprint": fingerprint},
        )
    )
    return brief, True


def context_status_payload(
    snapshot: EventContextSnapshot,
    *,
    bootstrap: dict[str, Any] | None = None,
) -> dict[str, Any]:
    readiness = dict(snapshot.pack.get("readiness") or {})
    brief = snapshot.pack.get("active_brief")
    return {
        "snapshot_id": snapshot.id,
        "snapshot_version": snapshot.version,
        "fingerprint": snapshot.fingerprint,
        "readiness": readiness,
        "provider_cache_allowed": snapshot.provider_cache_allowed,
        "active_brief": brief,
        "bootstrap": bootstrap,
        "created_at": _utc(snapshot.created_at),
    }


__all__ = [
    "ContextReadiness",
    "EventBriefPayload",
    "EventContextPack",
    "EventEvidenceRef",
    "GroundedText",
    "context_status_payload",
    "publish_event_brief",
    "refresh_event_context_snapshot",
    "render_event_bootstrap_input",
    "render_event_context_pack",
]
