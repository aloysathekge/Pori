"""Durable files: downloads + the user file library.

The library is the personal-OS pattern: a file saved here gets a
KnowledgeEntry pointer so every future run KNOWS it exists, and the
fetch_my_file tool can pull it into any conversation's sandbox."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from ..config import settings
from ..database import get_session
from ..library import add_to_library, remove_from_library
from ..models import StoredFile
from ..storage import get_object_store
from ..tenancy import OrganizationContext, Permission, require_permission

router = APIRouter(prefix="/files", tags=["files"])

_CHUNK = 256 * 1024


def _file_view(r: StoredFile) -> dict:
    return {
        "file_id": r.id,
        "name": r.name,
        "size_bytes": r.size_bytes,
        "content_type": r.content_type,
        "kind": r.kind,
        "in_library": r.in_library,
        "conversation_id": r.conversation_id,
        "created_at": r.created_at,
    }


@router.get("")
async def list_my_library(
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
):
    """The caller's file library (their own saved files, newest first)."""
    rows = (
        (
            await session.execute(
                select(StoredFile)
                .where(
                    StoredFile.organization_id == context.organization_id,
                    StoredFile.user_id == context.user_id,
                    StoredFile.in_library == True,  # noqa: E712
                )
                .order_by(col(StoredFile.created_at).desc())
            )
        )
        .scalars()
        .all()
    )
    return [_file_view(r) for r in rows]


async def _own_file(session, context, file_id: str) -> StoredFile:
    record = await session.get(StoredFile, file_id)
    if (
        record is None
        or record.organization_id != context.organization_id
        or record.user_id != context.user_id  # the library is personal
    ):
        raise HTTPException(status_code=404, detail="File not found")
    return record


@router.post("/{file_id}/library")
async def save_to_library(
    file_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
):
    """Save a file to the caller's library: flags the row + writes the memory
    pointer (one transaction), so every future run knows the file exists."""
    record = await _own_file(session, context, file_id)
    await add_to_library(session, record)
    await session.commit()
    return _file_view(record)


@router.delete("/{file_id}/library")
async def remove_from_my_library(
    file_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
):
    """Remove from the library: unflags + deletes the memory pointer in the
    same transaction — memory never points at nothing. The file itself (and
    its conversation history) remains."""
    record = await _own_file(session, context, file_id)
    await remove_from_library(session, record)
    await session.commit()
    return _file_view(record)


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
    presigned = store.url(
        record.storage_key, expires_s=settings.storage_presign_expiry_seconds
    )
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
