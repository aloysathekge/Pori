"""Conversation artifact endpoints.

Contract: expose files the conversation's agent runs wrote — the allowlist
comes from message receipts (``_conversation_artifacts``), and content is
served from the object store (durable across replicas/redeploys), capped at
``_ARTIFACT_MAX_BYTES``.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select
from starlette.concurrency import run_in_threadpool

from ...database import get_session
from ...file_presentations import presentation_kind
from ...models import Message, StoredFile
from ...storage import get_object_store
from ...tenancy import OrganizationContext, Permission, require_permission
from ._helpers import (
    _ARTIFACT_MAX_BYTES,
    _LANG_BY_EXT,
    _conversation_artifacts,
    _load_conv,
)

router = APIRouter()


@router.get("/{conversation_id}/artifacts")
async def list_artifacts(
    conversation_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """Files the conversation's agent runs wrote (from message receipts)."""
    await _load_conv(session, context, conversation_id)
    msgs = (
        (
            await session.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(col(Message.created_at))
            )
        )
        .scalars()
        .all()
    )
    return list(_conversation_artifacts(msgs).values())


def _read_artifact_head(storage_key: str) -> bytes:
    """Blocking object-store read (first ``_ARTIFACT_MAX_BYTES`` + 1 bytes)."""
    with get_object_store().open(storage_key) as fh:
        return fh.read(_ARTIFACT_MAX_BYTES + 1)


@router.get("/{conversation_id}/artifacts/content")
async def get_artifact_content(
    conversation_id: str,
    path: str = Query(..., max_length=1024),
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Read one artifact's text, allowlisted to paths this conversation wrote
    and served from the object store (durable across replicas/redeploys)."""
    await _load_conv(session, context, conversation_id)
    msgs = (
        (
            await session.execute(
                select(Message).where(Message.conversation_id == conversation_id)
            )
        )
        .scalars()
        .all()
    )
    entry = _conversation_artifacts(msgs).get(path)
    if entry is None:
        raise HTTPException(
            status_code=404, detail="Not an artifact of this conversation"
        )
    file_id = entry.get("file_id")
    record = await session.get(StoredFile, file_id) if file_id else None
    if (
        record is None
        or record.organization_id != context.organization_id
        or record.conversation_id != conversation_id
    ):
        raise HTTPException(status_code=404, detail="Artifact file no longer available")
    suffix = PurePosixPath(record.name).suffix.lower()
    renderer = presentation_kind(record.name, record.content_type)
    content = None
    truncated = False
    if renderer in {"code", "html", "markdown", "text"}:
        try:
            raw = await run_in_threadpool(_read_artifact_head, record.storage_key)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail="Artifact file no longer available"
            )
        truncated = len(raw) > _ARTIFACT_MAX_BYTES
        content = raw[:_ARTIFACT_MAX_BYTES].decode("utf-8", errors="replace")
    return {
        "path": path,
        "file_id": record.id,
        "name": record.name,
        "size_bytes": record.size_bytes,
        "content_type": record.content_type,
        "kind": record.kind,
        "event_id": record.event_id,
        "conversation_id": record.conversation_id,
        "created_at": record.created_at,
        "renderer": renderer,
        "content": content,
        "language": _LANG_BY_EXT.get(suffix, "text"),
        "truncated": truncated,
    }
