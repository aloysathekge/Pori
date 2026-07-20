"""Event-scoped evidence and canonical record tools for sourced research.

The Pori kernel retrieves provider-neutral web observations. Aloy owns their
durability and tenancy: every observation is committed to Event memory before
its evidence id is returned to the model, and structured records can cite only
evidence from the same Event.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from pori import RunContext
from pori.tools.standard.internet_tools import (
    WEB_EVIDENCE_CONTEXT_KEY,
    WEB_PAGE_READER_CONTEXT_KEY,
)

from ..context_ingestion import ContextIngestionError, fetch_public_link
from ..database import async_session
from ..models import Event, EventTrailEntry, KnowledgeEntry

EVENT_RECORD_HANDLER_CONTEXT_KEY = "event_record_handler"
EVENT_RECORD_TOOL_NAMES = frozenset({"event_record_upsert", "event_records_list"})

_NAMESPACE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_RECORD_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,199}$")
_MAX_OBSERVATION_EXCERPT = 16_000
_MAX_RECORD_DATA_CHARS = 50_000


class EventWebPageReader:
    """Use Aloy's SSRF-safe, redirect-bounded public fetcher for web research."""

    async def read(self, url: str, max_chars: int) -> dict[str, Any]:
        try:
            source = await fetch_public_link(url.strip())
        except ContextIngestionError as exc:
            return {
                "success": False,
                "status": "unavailable" if not exc.retryable else "failed",
                "error": str(exc),
                "retryable": exc.retryable,
                "url": url,
            }
        content = source.text[:max_chars]
        final_url = str(source.metadata.get("final_url") or url)
        return {
            "success": True,
            "status": "completed",
            "url": final_url,
            "title": source.title or final_url,
            "content": content,
            "truncated": len(source.text) > max_chars,
            "retrieved_at": source.retrieved_at.isoformat(),
            "evidence": {
                "kind": "web_page",
                "url": final_url,
                "title": source.title or final_url,
                "retrieved_at": source.retrieved_at.isoformat(),
                "provider": "direct",
                "content_sha256": source.sha256,
            },
        }


def _utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class WebObservation(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    kind: Literal["web_search_result", "web_page"]
    url: str = Field(min_length=8, max_length=5000)
    title: str = Field(min_length=1, max_length=1000)
    retrieved_at: datetime
    provider: str = Field(min_length=1, max_length=100)
    query: str | None = Field(default=None, max_length=5000)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    excerpt: str = Field(default="", max_length=100_000)

    @field_validator("url")
    @classmethod
    def _public_web_url(cls, value: str) -> str:
        if not value.lower().startswith(("https://", "http://")):
            raise ValueError("Evidence URL must use HTTP or HTTPS")
        return value


class EventEvidenceRecorder:
    """Commit immutable web observations on the database engine's owner loop."""

    def __init__(
        self,
        *,
        run_context: RunContext,
        task_id: str | None = None,
        session_factory: Any = async_session,
        owner_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._run_context = run_context
        self._task_id = task_id
        self._session_factory = session_factory
        self._owner_loop = owner_loop
        self._ids: list[str] = []
        self._lock = threading.Lock()

    async def _on_owner_loop(self, coroutine):
        current = asyncio.get_running_loop()
        if self._owner_loop is None or self._owner_loop is current:
            return await coroutine
        future = asyncio.run_coroutine_threadsafe(coroutine, self._owner_loop)
        return await asyncio.wrap_future(future)

    async def _load_event(self, session: AsyncSession) -> Event:
        event_id = self._run_context.event_id
        if not event_id:
            raise ValueError("Event identity is required for evidence persistence")
        event = await session.get(Event, event_id)
        if (
            event is None
            or event.organization_id != self._run_context.organization_id
            or event.user_id != self._run_context.user_id
        ):
            raise ValueError("Event is unavailable")
        if event.lifecycle == "archived":
            raise ValueError("Event is archived")
        return event

    def _evidence_id(self, observation: WebObservation) -> str:
        identity = "\x1f".join(
            [
                self._run_context.organization_id,
                self._run_context.event_id or "",
                self._run_context.run_id,
                observation.kind,
                observation.url,
                observation.content_sha256,
            ]
        )
        return f"evd_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]}"

    async def _record_many(
        self, observations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        parsed = [WebObservation.model_validate(item) for item in observations]
        if not parsed:
            return []
        async with self._session_factory() as session:
            event = await self._load_event(session)
            receipts: list[dict[str, Any]] = []
            added: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for observation in parsed:
                evidence_id = self._evidence_id(observation)
                first_occurrence = evidence_id not in seen_ids
                seen_ids.add(evidence_id)
                if (
                    first_occurrence
                    and await session.get(KnowledgeEntry, evidence_id) is None
                ):
                    metadata = observation.model_dump(mode="json", exclude={"excerpt"})
                    metadata["record_type"] = "web_evidence"
                    metadata["task_id"] = self._task_id
                    excerpt = observation.excerpt[:_MAX_OBSERVATION_EXCERPT]
                    session.add(
                        KnowledgeEntry(
                            id=evidence_id,
                            organization_id=event.organization_id,
                            user_id=event.user_id,
                            event_id=event.id,
                            agent_id=self._run_context.agent_id,
                            session_id=self._run_context.session_id,
                            content=(
                                f"{observation.title}\n{observation.url}"
                                + (f"\n\n{excerpt}" if excerpt else "")
                            ),
                            tags=["web_evidence", f"web:{observation.kind}"],
                            importance=2,
                            kind="episodic",
                            confidence=1.0,
                            sensitivity="internal",
                            source="web_tool",
                            provenance={
                                "source": "web_tool",
                                "source_id": observation.url,
                                "actor_id": self._run_context.agent_id,
                                "conversation_id": self._run_context.session_id,
                                "run_id": self._run_context.run_id,
                                "metadata": {
                                    "provider": observation.provider,
                                    "query": observation.query,
                                },
                            },
                            metadata_=metadata,
                            event_at=_utc(observation.retrieved_at),
                        )
                    )
                    added.append(
                        {
                            "evidence_id": evidence_id,
                            "url": observation.url,
                            "title": observation.title,
                            "retrieved_at": _utc(observation.retrieved_at).isoformat(),
                        }
                    )
                receipts.append(
                    {
                        "evidence_id": evidence_id,
                        "persisted": True,
                        "event_id": event.id,
                    }
                )
            if added:
                session.add(
                    EventTrailEntry(
                        organization_id=event.organization_id,
                        user_id=event.user_id,
                        event_id=event.id,
                        actor_id=self._run_context.agent_id,
                        kind="research_evidence_added",
                        summary=f"Recorded {len(added)} sourced web observation(s)",
                        run_id=self._run_context.run_id,
                        task_id=self._task_id,
                        evidence_refs=added,
                        payload={"count": len(added)},
                    )
                )
            await session.commit()
        with self._lock:
            self._ids.extend(receipt["evidence_id"] for receipt in receipts)
        return receipts

    async def record_many(
        self, observations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return await self._on_owner_loop(self._record_many(observations))

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(dict.fromkeys(self._ids))


class EventRecordUpsertParams(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    namespace: str = Field(min_length=1, max_length=64)
    record_key: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=1000)
    summary: str = Field(default="", max_length=10_000)
    data: dict[str, Any] = Field(default_factory=dict)
    posture: Literal["observed", "inferred", "unverified"] = "observed"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("namespace")
    @classmethod
    def _valid_namespace(cls, value: str) -> str:
        if not _NAMESPACE.fullmatch(value):
            raise ValueError("Invalid Event record namespace")
        return value

    @field_validator("record_key")
    @classmethod
    def _valid_record_key(cls, value: str) -> str:
        if not _RECORD_KEY.fullmatch(value):
            raise ValueError("Invalid Event record key")
        return value

    @field_validator("data")
    @classmethod
    def _bounded_data(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(value, default=str, sort_keys=True)) > _MAX_RECORD_DATA_CHARS:
            raise ValueError("Event record data is too large")
        return value

    @model_validator(mode="after")
    def _evidence_required(self) -> "EventRecordUpsertParams":
        if self.posture in {"observed", "inferred"} and not self.evidence_ids:
            raise ValueError(f"{self.posture} records require source evidence")
        return self


class EventRecordsListParams(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    namespace: str | None = Field(default=None, max_length=64)
    limit: int = Field(default=100, ge=1, le=500)

    @field_validator("namespace")
    @classmethod
    def _valid_namespace(cls, value: str | None) -> str | None:
        if value is not None and not _NAMESPACE.fullmatch(value):
            raise ValueError("Invalid Event record namespace")
        return value


def event_record_payload(row: KnowledgeEntry) -> dict[str, Any]:
    metadata = row.metadata_ or {}
    return {
        "id": row.id,
        "namespace": metadata.get("namespace"),
        "key": metadata.get("record_key"),
        "title": metadata.get("title") or row.content.splitlines()[0],
        "summary": metadata.get("summary") or "",
        "data": metadata.get("data") or {},
        "posture": metadata.get("posture") or "unverified",
        "confidence": row.confidence,
        "revision": int(metadata.get("revision") or 1),
        "evidence_refs": list(metadata.get("evidence_refs") or []),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


class EventRecordHandler:
    """Version generic Event records and enforce same-Event evidence references."""

    def __init__(
        self,
        *,
        run_context: RunContext,
        task_id: str | None = None,
        session_factory: Any = async_session,
        owner_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._run_context = run_context
        self._task_id = task_id
        self._session_factory = session_factory
        self._owner_loop = owner_loop
        self._record_ids: list[str] = []
        self._lock = threading.Lock()

    async def _on_owner_loop(self, coroutine):
        current = asyncio.get_running_loop()
        if self._owner_loop is None or self._owner_loop is current:
            return await coroutine
        future = asyncio.run_coroutine_threadsafe(coroutine, self._owner_loop)
        return await asyncio.wrap_future(future)

    async def _load_event(self, session: AsyncSession) -> Event:
        event_id = self._run_context.event_id
        if not event_id:
            raise ValueError("Event identity is required")
        event = await session.get(Event, event_id)
        if (
            event is None
            or event.organization_id != self._run_context.organization_id
            or event.user_id != self._run_context.user_id
        ):
            raise ValueError("Event is unavailable")
        if event.lifecycle == "archived":
            raise ValueError("Event is archived")
        return event

    async def _validated_evidence(
        self,
        session: AsyncSession,
        *,
        event: Event,
        evidence_ids: list[str],
    ) -> list[dict[str, Any]]:
        if not evidence_ids:
            return []
        unique_ids = list(dict.fromkeys(evidence_ids))
        rows = list(
            (
                await session.execute(
                    select(KnowledgeEntry).where(
                        col(KnowledgeEntry.id).in_(unique_ids),
                        KnowledgeEntry.organization_id == event.organization_id,
                        KnowledgeEntry.user_id == event.user_id,
                        KnowledgeEntry.event_id == event.id,
                        KnowledgeEntry.status == "active",
                    )
                )
            )
            .scalars()
            .all()
        )
        evidence = {row.id: row for row in rows if "web_evidence" in (row.tags or [])}
        missing = [
            evidence_id for evidence_id in unique_ids if evidence_id not in evidence
        ]
        if missing:
            raise ValueError(
                "Evidence is unavailable in this Event: " + ", ".join(missing)
            )
        refs: list[dict[str, Any]] = []
        for evidence_id in unique_ids:
            row = evidence[evidence_id]
            metadata = row.metadata_ or {}
            refs.append(
                {
                    "evidence_id": row.id,
                    "url": metadata.get("url"),
                    "title": metadata.get("title"),
                    "retrieved_at": metadata.get("retrieved_at"),
                }
            )
        return refs

    async def _upsert(self, params: EventRecordUpsertParams) -> dict[str, Any]:
        async with self._session_factory() as session:
            event = await self._load_event(session)
            evidence_refs = await self._validated_evidence(
                session, event=event, evidence_ids=params.evidence_ids
            )
            conflict_key = f"event_record:{params.namespace}:{params.record_key}"
            request_fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "organization_id": event.organization_id,
                        "event_id": event.id,
                        "run_id": self._run_context.run_id,
                        "namespace": params.namespace,
                        "record_key": params.record_key,
                        "title": params.title,
                        "summary": params.summary,
                        "data": params.data,
                        "posture": params.posture,
                        "confidence": params.confidence,
                        "evidence_ids": sorted(set(params.evidence_ids)),
                    },
                    default=str,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            record_id = f"erec_{request_fingerprint[:32]}"
            duplicate = await session.get(KnowledgeEntry, record_id)
            if duplicate is not None:
                if (
                    duplicate.organization_id != event.organization_id
                    or duplicate.user_id != event.user_id
                    or duplicate.event_id != event.id
                ):
                    raise ValueError("Event record idempotency conflict")
                with self._lock:
                    self._record_ids.append(duplicate.id)
                return event_record_payload(duplicate)
            existing = (
                (
                    await session.execute(
                        select(KnowledgeEntry)
                        .where(
                            KnowledgeEntry.organization_id == event.organization_id,
                            KnowledgeEntry.user_id == event.user_id,
                            KnowledgeEntry.event_id == event.id,
                            KnowledgeEntry.conflict_key == conflict_key,
                            KnowledgeEntry.status == "active",
                        )
                        .order_by(col(KnowledgeEntry.updated_at).desc())
                    )
                )
                .scalars()
                .first()
            )
            revision = (
                int(
                    ((existing.metadata_ if existing else {}) or {}).get("revision")
                    or 0
                )
                + 1
            )
            now = datetime.now(timezone.utc)
            row = KnowledgeEntry(
                id=record_id,
                organization_id=event.organization_id,
                user_id=event.user_id,
                event_id=event.id,
                agent_id=self._run_context.agent_id,
                session_id=self._run_context.session_id,
                content=params.title
                + (f"\n\n{params.summary}" if params.summary else ""),
                tags=["event_record", f"event_record:{params.namespace}"],
                importance=3,
                kind="semantic",
                confidence=params.confidence,
                sensitivity="internal",
                source="agent",
                provenance={
                    "source": "agent",
                    "source_id": record_id,
                    "actor_id": self._run_context.agent_id,
                    "conversation_id": self._run_context.session_id,
                    "run_id": self._run_context.run_id,
                    "metadata": {"task_id": self._task_id},
                },
                conflict_key=conflict_key,
                metadata_={
                    "record_type": "event_record",
                    "namespace": params.namespace,
                    "record_key": params.record_key,
                    "title": params.title,
                    "summary": params.summary,
                    "data": params.data,
                    "posture": params.posture,
                    "revision": revision,
                    "evidence_refs": evidence_refs,
                    "task_id": self._task_id,
                    "run_id": self._run_context.run_id,
                },
                event_at=now,
            )
            if existing is not None:
                existing.status = "superseded"
                existing.superseded_by = row.id
                existing.updated_at = now
                session.add(existing)
            session.add(row)
            session.add(
                EventTrailEntry(
                    organization_id=event.organization_id,
                    user_id=event.user_id,
                    event_id=event.id,
                    actor_id=self._run_context.agent_id,
                    kind="event_record_changed",
                    summary=f"Updated {params.namespace} record {params.title}",
                    run_id=self._run_context.run_id,
                    task_id=self._task_id,
                    evidence_refs=evidence_refs,
                    payload={
                        "record_id": row.id,
                        "namespace": params.namespace,
                        "record_key": params.record_key,
                        "revision": revision,
                        "posture": params.posture,
                    },
                )
            )
            event.updated_at = now
            session.add(event)
            await session.commit()
            await session.refresh(row)
        with self._lock:
            self._record_ids.append(row.id)
        return event_record_payload(row)

    async def upsert(self, params: EventRecordUpsertParams) -> dict[str, Any]:
        return await self._on_owner_loop(self._upsert(params))

    async def _list(self, params: EventRecordsListParams) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            event = await self._load_event(session)
            rows = list(
                (
                    await session.execute(
                        select(KnowledgeEntry)
                        .where(
                            KnowledgeEntry.organization_id == event.organization_id,
                            KnowledgeEntry.user_id == event.user_id,
                            KnowledgeEntry.event_id == event.id,
                            KnowledgeEntry.status == "active",
                        )
                        .order_by(col(KnowledgeEntry.updated_at).desc())
                    )
                )
                .scalars()
                .all()
            )
        records = [
            row
            for row in rows
            if (row.metadata_ or {}).get("record_type") == "event_record"
            and (
                params.namespace is None
                or (row.metadata_ or {}).get("namespace") == params.namespace
            )
        ]
        return [event_record_payload(row) for row in records[: params.limit]]

    async def list(self, params: EventRecordsListParams) -> list[dict[str, Any]]:
        return await self._on_owner_loop(self._list(params))

    @property
    def record_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(dict.fromkeys(self._record_ids))


async def event_record_upsert_tool(
    params: EventRecordUpsertParams, context: dict[str, Any]
) -> dict[str, Any]:
    handler = context.get(EVENT_RECORD_HANDLER_CONTEXT_KEY)
    if not isinstance(handler, EventRecordHandler):
        raise ValueError("Event record persistence is unavailable for this run")
    return await handler.upsert(params)


async def event_records_list_tool(
    params: EventRecordsListParams, context: dict[str, Any]
) -> list[dict[str, Any]]:
    handler = context.get(EVENT_RECORD_HANDLER_CONTEXT_KEY)
    if not isinstance(handler, EventRecordHandler):
        raise ValueError("Event record access is unavailable for this run")
    return await handler.list(params)


def register_research_tools(registry) -> None:
    if "event_record_upsert" not in registry.tools:
        registry.register_tool(
            name="event_record_upsert",
            param_model=EventRecordUpsertParams,
            function=event_record_upsert_tool,
            description=(
                "Create or revise one generic canonical Event record. Observed or "
                "inferred fields must cite evidence_ids returned by web_search or "
                "read_web_page. Use posture=unverified for unsupported facts instead "
                "of guessing. This is durable data for memory and Surfaces, not a Task."
            ),
        )
    if "event_records_list" not in registry.tools:
        registry.register_tool(
            name="event_records_list",
            param_model=EventRecordsListParams,
            function=event_records_list_tool,
            description=(
                "List current canonical records in this Event, optionally by namespace."
            ),
        )


__all__ = [
    "EVENT_RECORD_HANDLER_CONTEXT_KEY",
    "EVENT_RECORD_TOOL_NAMES",
    "EventEvidenceRecorder",
    "EventWebPageReader",
    "EventRecordHandler",
    "EventRecordUpsertParams",
    "EventRecordsListParams",
    "WEB_EVIDENCE_CONTEXT_KEY",
    "WEB_PAGE_READER_CONTEXT_KEY",
    "event_record_payload",
    "register_research_tools",
]
