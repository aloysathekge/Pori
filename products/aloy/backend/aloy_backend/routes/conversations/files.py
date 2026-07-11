"""Durable conversation file uploads.

Contract: rung 4 of the attachment ladder — beyond the inline/native limits,
the model gets a *reference* and works on the bytes in the sandbox. Enforces
per-file size and org storage quotas, stores the blob durably, and eagerly
provisions it into the conversation's sandbox so send time pays no copy.
"""

from __future__ import annotations

import hashlib
import logging
from typing import BinaryIO

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from starlette.concurrency import run_in_threadpool

from ...config import settings
from ...database import get_session
from ...models import StoredFile
from ...provisioning import provision_conversation_uploads
from ...storage import get_object_store, safe_name, upload_key
from ...tenancy import OrganizationContext, Permission, require_permission
from ._helpers import _load_conv

logger = logging.getLogger("aloy_backend")

router = APIRouter()


def _sha256_hexdigest(body: BinaryIO) -> str:
    """Blocking: hash the (already fully received) spooled body, then rewind."""
    digest = hashlib.sha256()
    for chunk in iter(lambda: body.read(1024 * 1024), b""):
        digest.update(chunk)
    body.seek(0)
    return digest.hexdigest()


@router.post("/{conversation_id}/files", status_code=201)
async def upload_conversation_file(
    conversation_id: str,
    file: UploadFile = File(...),
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
):
    """Store a large attachment durably (rung 4 of the attachment ladder:
    beyond the inline/native limits, the model gets a *reference* and works
    on the bytes in the sandbox). Also eagerly provisions the file into this
    conversation's sandbox so send time pays no copy."""
    conv = await _load_conv(session, context, conversation_id)

    body = file.file  # spooled temp file — already fully received
    body.seek(0, 2)
    size = body.tell()
    body.seek(0)
    logger.info(
        "Upload received: %r (%d bytes) for conversation %s",
        file.filename,
        size,
        conv.id,
    )
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
        conversation_id=conv.id,
        kind="upload",
        name=name,
        content_type=content_type,
        size_bytes=size,
        sha256=sha256,
        storage_key="",
    )
    record.storage_key = upload_key(context.organization_id, conv.id, record.id, name)
    await run_in_threadpool(
        get_object_store().put, record.storage_key, body, content_type=content_type
    )
    session.add(record)
    await session.commit()

    # Eager provisioning (latency contract: the copy happens NOW, while the
    # user is still typing — run setup just verifies the hash manifest).
    try:
        provision_conversation_uploads(conv.id, [record])
    except Exception:
        logger.exception("Eager provisioning failed for upload %s", record.id)

    return {
        "file_id": record.id,
        "name": record.name,
        "size_bytes": record.size_bytes,
        "content_type": record.content_type,
    }
