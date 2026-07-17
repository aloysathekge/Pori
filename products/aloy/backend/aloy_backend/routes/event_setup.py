"""Durable, resumable Event setup and one-time promotion into an Event."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select
from starlette.concurrency import run_in_threadpool

from ..config import settings
from ..database import get_session
from ..event_presenters import context_item_payload, event_payload
from ..events import ensure_event_conversation
from ..models import (
    ORG_CONNECTION_USER,
    Event,
    EventConnectionGrant,
    EventSetupContextItem,
    EventSetupDraft,
    EventTrailEntry,
    KnowledgeEntry,
    OAuthConnection,
    StoredFile,
)
from ..rate_limit import rate_limited_permission
from ..storage import event_setup_file_key, get_object_store, safe_name
from ..tenancy import OrganizationContext, Permission

router = APIRouter(prefix="/event-drafts", tags=["event-setup"])

SetupMode = Literal["simple", "assisted"]
ContextKind = Literal["link", "connection", "template"]


class DraftCreateBody(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    mode: SetupMode = "simple"
    title: str = Field(default="", max_length=300)
    description: str = Field(default="", max_length=50_000)


class DraftUpdateBody(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    mode: SetupMode | None = None
    title: str | None = Field(default=None, max_length=300)
    description: str | None = Field(default=None, max_length=50_000)


class ContextCreateBody(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    kind: ContextKind
    label: str = Field(default="", max_length=500)
    url: HttpUrl | None = None
    provider: str | None = Field(default=None, max_length=100)
    connection_scope: Literal["user", "org"] = "user"
    access_scope: dict[str, Any] = Field(default_factory=dict)
    template_id: str | None = Field(default=None, max_length=200)


def _utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


async def _draft_payload(
    session: AsyncSession, draft: EventSetupDraft
) -> dict[str, Any]:
    items = list(
        (
            await session.execute(
                select(EventSetupContextItem)
                .where(EventSetupContextItem.draft_id == draft.id)
                .order_by(col(EventSetupContextItem.created_at))
            )
        )
        .scalars()
        .all()
    )
    return {
        "id": draft.id,
        "mode": draft.mode,
        "status": draft.status,
        "title": draft.title,
        "description": draft.description,
        "created_event_id": draft.created_event_id,
        "context_items": [context_item_payload(item) for item in items],
        "created_at": _utc(draft.created_at),
        "updated_at": _utc(draft.updated_at),
    }


async def _load_draft(
    session: AsyncSession,
    context: OrganizationContext,
    draft_id: str,
    *,
    lock: bool = False,
) -> EventSetupDraft:
    statement = select(EventSetupDraft).where(
        EventSetupDraft.id == draft_id,
        EventSetupDraft.organization_id == context.organization_id,
        EventSetupDraft.user_id == context.user_id,
    )
    if lock:
        statement = statement.with_for_update()
    draft = (await session.execute(statement)).scalars().first()
    if draft is None:
        raise HTTPException(status_code=404, detail="Event setup draft not found")
    return draft


def _require_open(draft: EventSetupDraft) -> None:
    if draft.status != "open":
        raise HTTPException(status_code=409, detail="Event setup draft is not open")


@router.post("", status_code=201)
async def create_draft(
    body: DraftCreateBody,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.AGENT_WRITE)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    existing = (
        (
            await session.execute(
                select(EventSetupDraft)
                .where(
                    EventSetupDraft.organization_id == context.organization_id,
                    EventSetupDraft.user_id == context.user_id,
                    EventSetupDraft.status == "open",
                )
                .order_by(col(EventSetupDraft.updated_at).desc())
            )
        )
        .scalars()
        .first()
    )
    if existing is not None:
        return await _draft_payload(session, existing)
    draft = EventSetupDraft(
        organization_id=context.organization_id,
        user_id=context.user_id,
        mode=body.mode,
        title=body.title,
        description=body.description,
    )
    session.add(draft)
    await session.commit()
    await session.refresh(draft)
    return await _draft_payload(session, draft)


@router.get("/current", response_model=None)
async def current_draft(
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.AGENT_READ)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any] | Response:
    draft = (
        (
            await session.execute(
                select(EventSetupDraft)
                .where(
                    EventSetupDraft.organization_id == context.organization_id,
                    EventSetupDraft.user_id == context.user_id,
                    EventSetupDraft.status == "open",
                )
                .order_by(col(EventSetupDraft.updated_at).desc())
            )
        )
        .scalars()
        .first()
    )
    if draft is None:
        return Response(status_code=204)
    return await _draft_payload(session, draft)


@router.get("/{draft_id}")
async def get_draft(
    draft_id: str,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.AGENT_READ)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    return await _draft_payload(session, await _load_draft(session, context, draft_id))


@router.patch("/{draft_id}")
async def update_draft(
    draft_id: str,
    body: DraftUpdateBody,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.AGENT_WRITE)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    draft = await _load_draft(session, context, draft_id)
    _require_open(draft)
    values = body.model_dump(exclude_unset=True)
    for key, value in values.items():
        setattr(draft, key, value)
    draft.updated_at = datetime.now(timezone.utc)
    session.add(draft)
    await session.commit()
    await session.refresh(draft)
    return await _draft_payload(session, draft)


async def _resolve_connection(
    session: AsyncSession,
    context: OrganizationContext,
    *,
    provider: str,
    scope: Literal["user", "org"],
) -> OAuthConnection:
    owner = context.user_id if scope == "user" else ORG_CONNECTION_USER
    connection = (
        (
            await session.execute(
                select(OAuthConnection).where(
                    OAuthConnection.organization_id == context.organization_id,
                    OAuthConnection.user_id == owner,
                    OAuthConnection.provider == provider,
                    OAuthConnection.scope == scope,
                    OAuthConnection.status == "active",
                )
            )
        )
        .scalars()
        .first()
    )
    if connection is None:
        raise HTTPException(status_code=404, detail="Active connection not found")
    return connection


@router.post("/{draft_id}/context", status_code=201)
async def add_context(
    draft_id: str,
    body: ContextCreateBody,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.AGENT_WRITE)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    draft = await _load_draft(session, context, draft_id)
    _require_open(draft)
    item = EventSetupContextItem(
        organization_id=context.organization_id,
        user_id=context.user_id,
        draft_id=draft.id,
        kind=body.kind,
        label=body.label,
    )
    if body.kind == "link":
        if body.url is None:
            raise HTTPException(status_code=422, detail="A link URL is required")
        source_url = str(body.url)
        item.source_url = source_url
        item.label = body.label or urlsplit(source_url).netloc
        item.content = source_url
        item.status = "pending"
        item.metadata_ = {"ingestion": "queued"}
    elif body.kind == "connection":
        if not body.provider:
            raise HTTPException(status_code=422, detail="A provider is required")
        connection = await _resolve_connection(
            session,
            context,
            provider=body.provider,
            scope=body.connection_scope,
        )
        duplicate = (
            (
                await session.execute(
                    select(EventSetupContextItem).where(
                        EventSetupContextItem.draft_id == draft.id,
                        EventSetupContextItem.connection_id == connection.id,
                    )
                )
            )
            .scalars()
            .first()
        )
        if duplicate is not None:
            return context_item_payload(duplicate)
        item.connection_id = connection.id
        item.label = body.label or connection.account_email or connection.provider
        item.status = "ready"
        item.access_scope = {
            "mode": "event",
            "resources": list(body.access_scope.get("resources") or [])[:100],
        }
        item.metadata_ = {
            "provider": connection.provider,
            "connection_scope": connection.scope,
            "account_email": connection.account_email,
        }
    else:
        if not body.template_id:
            raise HTTPException(status_code=422, detail="A template id is required")
        item.label = body.label or body.template_id
        item.content = body.template_id
        item.status = "ready"
        item.metadata_ = {"template_id": body.template_id}
    session.add(item)
    draft.updated_at = datetime.now(timezone.utc)
    session.add(draft)
    await session.commit()
    await session.refresh(item)
    return context_item_payload(item)


@router.post("/{draft_id}/files", status_code=201)
async def add_file(
    draft_id: str,
    file: UploadFile = File(...),
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.AGENT_WRITE)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    draft = await _load_draft(session, context, draft_id)
    _require_open(draft)
    body = file.file
    body.seek(0, 2)
    size = body.tell()
    body.seek(0)
    if size == 0:
        raise HTTPException(status_code=422, detail="Empty context file")
    if size > settings.storage_max_file_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Context file exceeds the limit")
    staged_bytes = int(
        (
            await session.execute(
                select(func.coalesce(func.sum(EventSetupContextItem.size_bytes), 0))
                .join(
                    EventSetupDraft,
                    col(EventSetupDraft.id) == col(EventSetupContextItem.draft_id),
                )
                .where(
                    EventSetupContextItem.organization_id == context.organization_id,
                    EventSetupDraft.status == "open",
                )
            )
        ).scalar_one()
    )
    stored_bytes = int(
        (
            await session.execute(
                select(func.coalesce(func.sum(StoredFile.size_bytes), 0)).where(
                    StoredFile.organization_id == context.organization_id
                )
            )
        ).scalar_one()
    )
    if staged_bytes + stored_bytes + size > settings.storage_org_quota_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail="Organization storage quota exceeded"
        )
    digest = hashlib.sha256()
    for chunk in iter(lambda: body.read(1024 * 1024), b""):
        digest.update(chunk)
    body.seek(0)
    name = safe_name(file.filename or "context-file")
    content_type = file.content_type or "application/octet-stream"
    item = EventSetupContextItem(
        organization_id=context.organization_id,
        user_id=context.user_id,
        draft_id=draft.id,
        kind="file",
        status="pending",
        label=name,
        content_type=content_type,
        size_bytes=size,
        sha256=digest.hexdigest(),
        metadata_={"ingestion": {"status": "queued"}},
    )
    item.storage_key = event_setup_file_key(
        context.organization_id, draft.id, item.id, name
    )
    await run_in_threadpool(
        get_object_store().put,
        item.storage_key,
        body,
        content_type=content_type,
    )
    session.add(item)
    draft.updated_at = datetime.now(timezone.utc)
    session.add(draft)
    await session.commit()
    await session.refresh(item)
    return context_item_payload(item)


@router.delete("/{draft_id}/context/{item_id}", status_code=204)
async def remove_context(
    draft_id: str,
    item_id: str,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.AGENT_WRITE)
    ),
    session: AsyncSession = Depends(get_session),
) -> Response:
    draft = await _load_draft(session, context, draft_id)
    _require_open(draft)
    item = await session.get(EventSetupContextItem, item_id)
    if (
        item is None
        or item.draft_id != draft.id
        or item.organization_id != context.organization_id
        or item.user_id != context.user_id
    ):
        raise HTTPException(status_code=404, detail="Context item not found")
    if item.storage_key:
        await run_in_threadpool(get_object_store().delete, item.storage_key)
    await session.delete(item)
    draft.updated_at = datetime.now(timezone.utc)
    session.add(draft)
    await session.commit()
    return Response(status_code=204)


@router.post("/{draft_id}/promote", status_code=201)
async def promote_draft(
    draft_id: str,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.AGENT_WRITE)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    draft = await _load_draft(session, context, draft_id, lock=True)
    if draft.created_event_id:
        existing = await session.get(Event, draft.created_event_id)
        if existing is not None:
            return event_payload(existing)
    _require_open(draft)
    title = draft.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="An Event name is required")
    items = list(
        (
            await session.execute(
                select(EventSetupContextItem).where(
                    EventSetupContextItem.draft_id == draft.id,
                    EventSetupContextItem.organization_id == context.organization_id,
                    EventSetupContextItem.user_id == context.user_id,
                )
            )
        )
        .scalars()
        .all()
    )
    event = Event(
        organization_id=context.organization_id,
        user_id=context.user_id,
        type="project",
        title=title,
        lifecycle="active",
        phase="planning",
        summary=draft.description.strip()[:2000],
        metadata_={
            "notes": draft.description,
            "setup": {
                "mode": draft.mode,
                "draft_id": draft.id,
                "context_count": len(items),
                "bootstrap_status": "queued" if items or draft.description else "idle",
            },
            "cover": {
                "status": "queued",
                "source": "automatic",
                "version": 0,
            },
        },
    )
    session.add(event)
    await session.flush()
    conversation = await ensure_event_conversation(session, event=event)
    evidence: list[dict[str, str]] = []
    if draft.description.strip():
        session.add(
            KnowledgeEntry(
                organization_id=context.organization_id,
                user_id=context.user_id,
                event_id=event.id,
                session_id=conversation.id,
                content=draft.description.strip(),
                tags=["event-setup", "description"],
                importance=3,
                source="user",
                provenance={"source": "event_setup", "draft_id": draft.id},
                scope_level="personal",
                metadata_={"event_scoped": True},
            )
        )
    for item in items:
        item.event_id = event.id
        evidence.append({"context_item_id": item.id, "kind": item.kind})
        if item.kind == "file" and item.storage_key:
            session.add(
                StoredFile(
                    organization_id=context.organization_id,
                    user_id=context.user_id,
                    event_id=event.id,
                    origin_session_id=conversation.id,
                    conversation_id=conversation.id,
                    kind="upload",
                    in_library=True,
                    name=item.label,
                    content_type=item.content_type or "application/octet-stream",
                    size_bytes=item.size_bytes,
                    sha256=item.sha256,
                    storage_key=item.storage_key,
                )
            )
        elif item.kind == "link" and item.source_url:
            entry = KnowledgeEntry(
                organization_id=context.organization_id,
                user_id=context.user_id,
                event_id=event.id,
                session_id=conversation.id,
                content=f"Event setup link: {item.source_url}",
                tags=["event-setup", "link"],
                importance=2,
                source="user",
                provenance={
                    "source": "event_setup",
                    "draft_id": draft.id,
                    "context_item_id": item.id,
                    "url": item.source_url,
                },
                scope_level="personal",
                metadata_={"event_scoped": True, "ingestion_status": item.status},
            )
            session.add(entry)
            await session.flush()
            item.knowledge_entry_id = entry.id
        elif item.kind == "connection" and item.connection_id:
            session.add(
                EventConnectionGrant(
                    organization_id=context.organization_id,
                    user_id=context.user_id,
                    event_id=event.id,
                    connection_id=item.connection_id,
                    provider=str(item.metadata_.get("provider") or "unknown"),
                    access_scope=item.access_scope,
                )
            )
        elif item.kind == "template":
            session.add(
                KnowledgeEntry(
                    organization_id=context.organization_id,
                    user_id=context.user_id,
                    event_id=event.id,
                    session_id=conversation.id,
                    content=f"Starting template: {item.content}",
                    tags=["event-setup", "template"],
                    importance=2,
                    source="user",
                    provenance={
                        "source": "event_setup",
                        "draft_id": draft.id,
                        "context_item_id": item.id,
                    },
                    scope_level="personal",
                    metadata_={"event_scoped": True},
                )
            )
        session.add(item)
    session.add(
        EventTrailEntry(
            organization_id=context.organization_id,
            user_id=context.user_id,
            event_id=event.id,
            actor_id=context.user_id,
            kind="event_created",
            summary=f"Created Project Event {event.title}",
            evidence_refs=evidence,
            payload={
                "type": "project",
                "lifecycle": "active",
                "setup_draft_id": draft.id,
                "context_count": len(items),
            },
        )
    )
    draft.status = "promoted"
    draft.created_event_id = event.id
    draft.updated_at = datetime.now(timezone.utc)
    session.add(draft)
    await session.commit()
    await session.refresh(event)
    return event_payload(event)
