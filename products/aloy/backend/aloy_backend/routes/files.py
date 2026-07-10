"""Durable file downloads — StoredFile pointer rows → object-store bytes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models import StoredFile
from ..storage import get_object_store
from ..tenancy import OrganizationContext, Permission, require_permission

router = APIRouter(prefix="/files", tags=["files"])

_CHUNK = 256 * 1024


@router.get("/{file_id}")
async def download_file(
    file_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
):
    """Stream one stored file (org-checked). Presigned redirect when the
    store supports it; otherwise streamed through the backend."""
    record = await session.get(StoredFile, file_id)
    if record is None or record.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="File not found")
    store = get_object_store()
    presigned = store.url(record.storage_key)
    if presigned:
        return RedirectResponse(presigned, status_code=307)
    try:
        handle = store.open(record.storage_key)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File content no longer available")

    def _iter():
        with handle:
            while True:
                chunk = handle.read(_CHUNK)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        _iter(),
        media_type=record.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{record.name}"',
            "Content-Length": str(record.size_bytes),
        },
    )
