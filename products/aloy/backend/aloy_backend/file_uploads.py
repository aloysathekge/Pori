"""Shared durable-upload service for conversations and the personal library."""

from __future__ import annotations

import hashlib
from typing import BinaryIO

from fastapi import HTTPException, UploadFile
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from starlette.concurrency import run_in_threadpool

from .config import settings
from .models import StoredFile
from .storage import get_object_store, safe_name, upload_key
from .tenancy import OrganizationContext


def _sha256_hexdigest(body: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: body.read(1024 * 1024), b""):
        digest.update(chunk)
    body.seek(0)
    return digest.hexdigest()


async def store_user_upload(
    session: AsyncSession,
    context: OrganizationContext,
    file: UploadFile,
    *,
    event_id: str,
    conversation_id: str | None,
) -> StoredFile:
    """Validate and stage one user-owned upload without committing it.

    Routes own transaction boundaries so a library pointer or other related
    state can commit atomically with the blob pointer row.
    """
    body = file.file
    body.seek(0, 2)
    size = body.tell()
    body.seek(0)
    if size == 0:
        raise HTTPException(status_code=422, detail="Empty file")
    if size > settings.storage_max_file_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.storage_max_file_mb}MB limit",
        )

    used = (
        await session.execute(
            select(func.coalesce(func.sum(StoredFile.size_bytes), 0)).where(
                StoredFile.organization_id == context.organization_id
            )
        )
    ).scalar_one()
    if used + size > settings.storage_org_quota_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Organization storage quota exceeded",
        )

    sha256 = await run_in_threadpool(_sha256_hexdigest, body)
    name = safe_name(file.filename or "upload")
    content_type = file.content_type or "application/octet-stream"
    record = StoredFile(
        organization_id=context.organization_id,
        user_id=context.user_id,
        event_id=event_id,
        origin_session_id=conversation_id,
        conversation_id=conversation_id,
        kind="upload",
        name=name,
        content_type=content_type,
        size_bytes=size,
        sha256=sha256,
        storage_key="",
    )
    storage_scope = conversation_id or f"library-{context.user_id}"
    record.storage_key = upload_key(
        context.organization_id,
        storage_scope,
        record.id,
        name,
    )
    await run_in_threadpool(
        get_object_store().put,
        record.storage_key,
        body,
        content_type=content_type,
    )
    session.add(record)
    return record


__all__ = ["store_user_upload"]
