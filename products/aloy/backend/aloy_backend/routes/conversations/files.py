"""Conversation-scoped file discovery and durable uploads."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from ...database import get_session
from ...file_uploads import store_user_upload
from ...models import Event, StoredFile
from ...provisioning import provision_event_uploads
from ...tenancy import OrganizationContext, Permission, require_permission
from ._helpers import _load_conv

logger = logging.getLogger("aloy_backend")

router = APIRouter()


def _reference_view(record: StoredFile, event_title: str) -> dict:
    return {
        "file_id": record.id,
        "name": record.name,
        "size_bytes": record.size_bytes,
        "content_type": record.content_type,
        "kind": record.kind,
        "event_id": record.event_id,
        "event_title": event_title,
        "conversation_id": record.conversation_id,
        "in_library": record.in_library,
        "created_at": record.created_at,
    }


@router.get("/{conversation_id}/files")
async def list_conversation_files(
    conversation_id: str,
    q: str = Query(default="", max_length=120),
    limit: int = Query(default=50, ge=1, le=100),
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """List files that may be explicitly attached to this Conversation.

    Dedicated Events expose their own retained files. Life is the deliberate
    cross-Event chooser and may expose every file owned by this user; selecting
    one grants context only to the current turn.
    """
    conv = await _load_conv(session, context, conversation_id)
    event = await session.get(Event, conv.event_id)
    if event is None:
        return []

    statement = select(StoredFile).where(
        StoredFile.organization_id == context.organization_id,
        StoredFile.user_id == context.user_id,
        col(StoredFile.kind).in_(["upload", "artifact"]),
    )
    if not event.is_life:
        statement = statement.where(StoredFile.event_id == event.id)
    query = q.strip()
    if query:
        statement = statement.where(col(StoredFile.name).ilike(f"%{query}%"))
    records = list(
        (
            await session.execute(
                statement.order_by(col(StoredFile.created_at).desc()).limit(limit)
            )
        )
        .scalars()
        .all()
    )
    event_ids = {record.event_id for record in records}
    events = (
        (await session.execute(select(Event).where(col(Event.id).in_(event_ids))))
        .scalars()
        .all()
        if event_ids
        else []
    )
    titles = {row.id: row.title for row in events}
    return [
        _reference_view(record, titles.get(record.event_id, "Event"))
        for record in records
    ]


@router.post("/{conversation_id}/files", status_code=201)
async def upload_conversation_file(
    conversation_id: str,
    file: UploadFile = File(...),
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Store and eagerly provision one attachment for this Event."""
    conv = await _load_conv(session, context, conversation_id)
    record = await store_user_upload(
        session,
        context,
        file,
        event_id=conv.event_id,
        conversation_id=conv.id,
    )
    await session.commit()

    try:
        provision_event_uploads(conv.event_id, [record])
    except Exception:
        logger.exception("Eager provisioning failed for upload %s", record.id)

    return {
        "file_id": record.id,
        "name": record.name,
        "size_bytes": record.size_bytes,
        "content_type": record.content_type,
    }
