"""Durable files: downloads + the user file library.

The library is the personal-OS pattern: a file saved here gets a
KnowledgeEntry pointer so every future run KNOWS it exists, and the
fetch_my_file tool can pull it into any conversation's sandbox."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select
from starlette.concurrency import run_in_threadpool

from ..config import settings
from ..database import get_session
from ..doc_extract import ExtractionError
from ..events import ensure_life_event
from ..file_presentations import build_office_preview, presentation_kind
from ..file_uploads import store_user_upload
from ..library import add_to_library, remove_from_library
from ..models import StoredFile
from ..storage import get_object_store
from ..tenancy import OrganizationContext, Permission, require_permission

router = APIRouter(prefix="/files", tags=["files"])

_CHUNK = 256 * 1024
_MAX_OFFICE_PREVIEW_READ = 25 * 1024 * 1024 + 1
_OFFICE_PRESENTATION_KINDS = {"document", "spreadsheet", "slides"}


def _file_view(r: StoredFile) -> dict:
    return {
        "file_id": r.id,
        "name": r.name,
        "size_bytes": r.size_bytes,
        "content_type": r.content_type,
        "kind": r.kind,
        "in_library": r.in_library,
        "event_id": r.event_id,
        "conversation_id": r.conversation_id,
        "created_at": r.created_at,
    }


@router.get("")
async def list_my_library(
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
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


@router.post("", status_code=201)
async def upload_library_file(
    file: UploadFile = File(...),
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Upload directly into My files without manufacturing a chat turn."""
    life = await ensure_life_event(
        session,
        organization_id=context.organization_id,
        user_id=context.user_id,
    )
    record = await store_user_upload(
        session,
        context,
        file,
        event_id=life.id,
        conversation_id=None,
    )
    await add_to_library(session, record)
    await session.commit()
    return _file_view(record)


async def _own_file(
    session: AsyncSession, context: OrganizationContext, file_id: str
) -> StoredFile:
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
) -> dict:
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
) -> dict:
    """Remove from the library: unflags + deletes the memory pointer in the
    same transaction — memory never points at nothing. The file itself (and
    its conversation history) remains."""
    record = await _own_file(session, context, file_id)
    await remove_from_library(session, record)
    await session.commit()
    return _file_view(record)


def _read_file_bytes(storage_key: str) -> bytes:
    with get_object_store().open(storage_key) as handle:
        return handle.read(_MAX_OFFICE_PREVIEW_READ)


@router.get("/{file_id}/presentation")
async def get_file_presentation(
    file_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Return one typed, inert presentation plan for the trusted host viewer."""
    record = await _own_file(session, context, file_id)
    renderer = presentation_kind(record.name, record.content_type)
    store = get_object_store()
    source_url = store.url(
        record.storage_key, expires_s=settings.storage_presign_expiry_seconds
    )
    preview = None
    preview_error = None
    if renderer in _OFFICE_PRESENTATION_KINDS:
        try:
            raw = await run_in_threadpool(_read_file_bytes, record.storage_key)
            preview = await run_in_threadpool(build_office_preview, renderer, raw)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail="File content no longer available"
            )
        except ExtractionError as exc:
            # Opening/downloading the immutable original remains possible when a
            # bounded preview cannot be produced (corrupt, encrypted, or large).
            preview_error = str(exc)
    return {
        **_file_view(record),
        "renderer": renderer,
        "source_url": source_url,
        "preview": preview,
        "preview_error": preview_error,
        "sha256": record.sha256,
    }


def _parse_byte_range(value: str | None, size: int) -> tuple[int, int] | None:
    if not value:
        return None
    if not value.startswith("bytes=") or "," in value:
        raise HTTPException(
            status_code=416,
            detail="Only one byte range is supported",
            headers={"Content-Range": f"bytes */{size}"},
        )
    start_raw, separator, end_raw = value[6:].partition("-")
    if not separator:
        raise HTTPException(status_code=416, detail="Invalid byte range")
    try:
        if not start_raw:
            length = int(end_raw)
            if length <= 0:
                raise ValueError
            start = max(size - length, 0)
            end = size - 1
        else:
            start = int(start_raw)
            end = int(end_raw) if end_raw else size - 1
    except ValueError:
        raise HTTPException(status_code=416, detail="Invalid byte range")
    if start < 0 or start >= size or end < start:
        raise HTTPException(
            status_code=416,
            detail="Byte range is outside the file",
            headers={"Content-Range": f"bytes */{size}"},
        )
    return start, min(end, size - 1)


@router.get("/{file_id}")
async def download_file(
    file_id: str,
    range_header: str | None = Header(default=None, alias="Range"),
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Stream one stored file (org-checked). Presigned redirect when the
    store supports it; otherwise streamed through the backend."""
    record = await _own_file(session, context, file_id)
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

    byte_range = _parse_byte_range(range_header, record.size_bytes)
    start, end = byte_range or (0, max(record.size_bytes - 1, 0))
    length = max(end - start + 1, 0)

    def _iter():
        with handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(_CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f'inline; filename="{record.name}"',
        "Content-Length": str(length),
        "X-Content-Type-Options": "nosniff",
    }
    if byte_range:
        headers["Content-Range"] = f"bytes {start}-{end}/{record.size_bytes}"
    return StreamingResponse(
        _iter(),
        status_code=206 if byte_range else 200,
        media_type=record.content_type,
        headers=headers,
    )
