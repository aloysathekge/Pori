"""The user file library: durable files the system ALWAYS knows about.

The Aloy knowledge-model pattern — memory is an index over durable things:
adding a file to the library writes a KnowledgeEntry POINTER (~20 tokens,
user-scoped, no agent/session binding so every future run of this user can
recall it). The bytes stay in the object store; the ``fetch_my_file`` tool
materializes them into the current conversation's sandbox on demand.

Lifecycle rule: the pointer and the flag move together — removing a file
from the library deletes its knowledge entry in the same transaction, so
memory never points at nothing.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from .models import KnowledgeEntry, StoredFile

_ENTRY_PREFIX = "libfile-"


def library_entry_id(file_id: str) -> str:
    """Deterministic KnowledgeEntry id — makes add/remove/update an upsert."""
    return f"{_ENTRY_PREFIX}{file_id}"


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)}B" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def _pointer_content(record: StoredFile) -> str:
    return (
        f'The user\'s file library contains "{record.name}" '
        f"({_human_size(record.size_bytes)}, {record.content_type}, "
        f"saved {record.created_at:%Y-%m-%d}). When a task needs it, retrieve "
        f"it into the workspace with the fetch_my_file tool (name: {record.name})."
    )


async def add_to_library(session: AsyncSession, record: StoredFile) -> None:
    """Flag the file + upsert its memory pointer (staged, caller commits)."""
    record.in_library = True
    session.add(record)
    entry_id = library_entry_id(record.id)
    existing = await session.get(KnowledgeEntry, entry_id)
    if existing is not None:
        existing.content = _pointer_content(record)
        existing.updated_at = datetime.now(timezone.utc)
        session.add(existing)
        return
    session.add(
        KnowledgeEntry(
            id=entry_id,
            organization_id=record.organization_id,
            user_id=record.user_id,
            # No agent/session binding: the pointer belongs to the USER, so
            # any conversation's run can recall it.
            agent_id=None,
            session_id=None,
            content=_pointer_content(record),
            tags=["file-library"],
            importance=2,
            kind="semantic",
            source="user",
            scope_level="personal",
        )
    )


async def remove_from_library(session: AsyncSession, record: StoredFile) -> None:
    """Unflag + delete the memory pointer (staged, caller commits)."""
    record.in_library = False
    session.add(record)
    entry = await session.get(KnowledgeEntry, library_entry_id(record.id))
    if entry is not None:
        await session.delete(entry)


def library_manifest(records: list[StoredFile]) -> list[dict]:
    """The per-run manifest handed to the fetch_my_file tool via context —
    everything the tool needs to materialize a file without a DB session."""
    return [
        {
            "file_id": r.id,
            "event_id": getattr(r, "event_id", None),
            "name": r.name,
            "size_bytes": r.size_bytes,
            "content_type": r.content_type,
            "sha256": r.sha256,
            "storage_key": r.storage_key,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]
