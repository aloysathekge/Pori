from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from pori import MemoryCatalog, MemoryStatus

from ..database import get_session
from ..memory_records import (
    apply_conflict_policy,
    create_record,
    record_response,
    record_to_row,
    request_scope,
    row_to_record,
    search_records,
)
from ..models import CoreMemoryBlock, KnowledgeEntry
from ..schemas import (
    CoreMemoryBlockResponse,
    CoreMemoryBlockUpdate,
    CoreMemoryResponse,
    KnowledgeEntryCreate,
    KnowledgeEntryResponse,
    KnowledgeSearchRequest,
    MemoryExportResponse,
)
from ..tenancy import OrganizationContext, Permission, require_permission

logger = logging.getLogger("aloy_backend")

router = APIRouter(prefix="/me/memory", tags=["memory"])

DEFAULT_LABELS = ("persona", "human", "notes")
DEFAULT_CHAR_LIMIT = 2000


# ---- helpers ----


async def _get_or_create_block(
    session: AsyncSession, context: OrganizationContext, label: str
) -> CoreMemoryBlock:
    """Get a core memory block, creating it if it doesn't exist."""
    result = await session.execute(
        select(CoreMemoryBlock).where(
            CoreMemoryBlock.organization_id == context.organization_id,
            CoreMemoryBlock.user_id == context.user_id,
            CoreMemoryBlock.label == label,
        )
    )
    block = result.scalars().first()
    if not block:
        block = CoreMemoryBlock(
            organization_id=context.organization_id,
            user_id=context.user_id,
            label=label,
            value="",
            char_limit=DEFAULT_CHAR_LIMIT,
        )
        session.add(block)
        await session.commit()
        await session.refresh(block)
    return block


# ---- Core Memory ----


@router.get("", response_model=CoreMemoryResponse)
async def get_core_memory(
    context: OrganizationContext = Depends(require_permission(Permission.MEMORY_READ)),
    session: AsyncSession = Depends(get_session),
):
    """View all CoreMemory blocks (persona, human, notes)."""
    blocks = []
    for label in DEFAULT_LABELS:
        block = await _get_or_create_block(session, context, label)
        blocks.append(
            CoreMemoryBlockResponse(
                label=block.label,
                value=block.value,
                limit=block.char_limit,
                read_only=False,
            )
        )
    return CoreMemoryResponse(blocks=blocks)


@router.delete("", status_code=204)
async def reset_memory(
    context: OrganizationContext = Depends(require_permission(Permission.MEMORY_WRITE)),
    session: AsyncSession = Depends(get_session),
):
    """Reset all persistent memory for the current user."""
    # Clear core memory blocks
    result = await session.execute(
        select(CoreMemoryBlock).where(
            CoreMemoryBlock.organization_id == context.organization_id,
            CoreMemoryBlock.user_id == context.user_id,
        )
    )
    for block in result.scalars().all():
        block.value = ""
        block.updated_at = datetime.now(timezone.utc)
        session.add(block)

    # Soft-delete long-term records so deletion remains auditable.
    result = await session.execute(
        select(KnowledgeEntry).where(
            KnowledgeEntry.organization_id == context.organization_id,
            KnowledgeEntry.user_id == context.user_id,
        )
    )
    entries = result.scalars().all()
    catalog = MemoryCatalog(row_to_record(entry) for entry in entries)
    try:
        for record in list(catalog.records()):
            if record.status != MemoryStatus.DELETED:
                catalog.delete(record.id, record.scope)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    by_id = {entry.id: entry for entry in entries}
    for record in catalog.records():
        record_to_row(record, by_id[record.id])
        session.add(by_id[record.id])

    await session.commit()
    logger.info("Memory reset for user %s", context.user_id)


# ---- Knowledge Entries (replaces archival) ----


@router.get("/knowledge", response_model=list[KnowledgeEntryResponse])
async def list_knowledge(
    context: OrganizationContext = Depends(require_permission(Permission.MEMORY_READ)),
    session: AsyncSession = Depends(get_session),
    limit: int = 50,
    offset: int = 0,
):
    """List knowledge entries."""
    result = await session.execute(
        select(KnowledgeEntry)
        .where(
            KnowledgeEntry.organization_id == context.organization_id,
            KnowledgeEntry.user_id == context.user_id,
        )
        .order_by(KnowledgeEntry.created_at.desc())
    )
    records = [row_to_record(entry) for entry in result.scalars().all()]
    records = [record for record in records if record.is_retrievable()]
    return [record_response(record) for record in records[offset : offset + limit]]


@router.post("/knowledge", response_model=KnowledgeEntryResponse, status_code=201)
async def create_knowledge_entry(
    body: KnowledgeEntryCreate,
    context: OrganizationContext = Depends(require_permission(Permission.MEMORY_WRITE)),
    session: AsyncSession = Depends(get_session),
):
    """Manually add a knowledge entry."""
    record = create_record(
        user_id=context.user_id,
        organization_id=context.organization_id,
        content=body.content,
        agent_id=body.agent_id,
        session_id=body.session_id,
        tags=body.tags,
        importance=body.importance,
        kind=body.kind,
        confidence=body.confidence,
        sensitivity=body.sensitivity,
        source=body.source,
        source_id=body.source_id,
        conversation_id=body.conversation_id,
        run_id=body.run_id,
        retention=(
            body.retention
            or (
                {"ttl_days": context.policy.memory_retention_days}
                if context.policy.memory_retention_days
                else {}
            )
        ),
        conflict_key=body.conflict_key,
        event_at=body.event_at,
    )
    result = await session.execute(
        select(KnowledgeEntry).where(
            KnowledgeEntry.organization_id == context.organization_id,
            KnowledgeEntry.user_id == context.user_id,
        )
    )
    existing_rows = result.scalars().all()
    existing_by_id = {entry.id: entry for entry in existing_rows}
    try:
        catalog = apply_conflict_policy(
            (row_to_record(entry) for entry in existing_rows),
            record,
            body.conflict_policy,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    for catalog_record in catalog.records():
        existing = existing_by_id.get(catalog_record.id)
        if existing is not None:
            record_to_row(catalog_record, existing)
            session.add(existing)
    session.add(record_to_row(record))
    await session.commit()
    return record_response(record)


@router.delete("/knowledge/{entry_id}", status_code=204)
async def delete_knowledge_entry(
    entry_id: str,
    hard: bool = False,
    context: OrganizationContext = Depends(require_permission(Permission.MEMORY_WRITE)),
    session: AsyncSession = Depends(get_session),
):
    """Delete a specific knowledge entry."""
    entry = await session.get(KnowledgeEntry, entry_id)
    if not entry or entry.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Entry not found")
    catalog = MemoryCatalog([row_to_record(entry)])
    try:
        deleted = catalog.delete(entry_id, row_to_record(entry).scope, hard=hard)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Entry not found")
    if hard:
        await session.delete(entry)
    else:
        record_to_row(catalog.records()[0], entry)
        session.add(entry)
    await session.commit()


# Keep archival endpoints as aliases for backward compatibility
@router.get("/archival", response_model=list[KnowledgeEntryResponse])
async def list_archival(
    context: OrganizationContext = Depends(require_permission(Permission.MEMORY_READ)),
    session: AsyncSession = Depends(get_session),
    limit: int = 50,
    offset: int = 0,
):
    """List knowledge entries (archival alias)."""
    return await list_knowledge(
        context=context, session=session, limit=limit, offset=offset
    )


@router.post("/archival/search", response_model=list[KnowledgeEntryResponse])
async def search_archival(
    body: KnowledgeSearchRequest,
    context: OrganizationContext = Depends(require_permission(Permission.MEMORY_READ)),
    session: AsyncSession = Depends(get_session),
):
    """Search knowledge entries by tag filter."""
    stmt = select(KnowledgeEntry).where(
        KnowledgeEntry.organization_id == context.organization_id,
        KnowledgeEntry.user_id == context.user_id,
    )
    result = await session.execute(stmt)
    hits = search_records(
        (row_to_record(entry) for entry in result.scalars().all()),
        scope=request_scope(
            context.user_id,
            organization_id=context.organization_id,
            agent_id=body.agent_id,
            session_id=body.session_id,
        ),
        query=body.query,
        k=body.k,
        kinds=body.kinds,
        tags=body.tags,
        min_score=body.min_score,
    )
    return [record_response(hit.record) for hit in hits]


@router.delete("/archival/{passage_id}", status_code=204)
async def delete_archival_passage(
    passage_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.MEMORY_WRITE)),
    session: AsyncSession = Depends(get_session),
):
    """Delete a knowledge entry (archival alias)."""
    return await delete_knowledge_entry(
        entry_id=passage_id, context=context, session=session
    )


@router.get("/export/all", response_model=MemoryExportResponse)
async def export_memory(
    context: OrganizationContext = Depends(require_permission(Permission.MEMORY_READ)),
    session: AsyncSession = Depends(get_session),
):
    """Export all active and superseded memory owned by the current tenant."""
    result = await session.execute(
        select(KnowledgeEntry).where(
            KnowledgeEntry.organization_id == context.organization_id,
            KnowledgeEntry.user_id == context.user_id,
        )
    )
    records = [row_to_record(entry) for entry in result.scalars().all()]
    return MemoryExportResponse(
        records=[
            record_response(record)
            for record in records
            if record.status != MemoryStatus.DELETED
        ],
        exported_at=datetime.now(timezone.utc),
    )


@router.post("/retention/prune")
async def prune_expired_memory(
    context: OrganizationContext = Depends(require_permission(Permission.MEMORY_WRITE)),
    session: AsyncSession = Depends(get_session),
):
    """Permanently remove expired records not protected by legal hold."""
    result = await session.execute(
        select(KnowledgeEntry).where(
            KnowledgeEntry.organization_id == context.organization_id,
            KnowledgeEntry.user_id == context.user_id,
        )
    )
    entries = result.scalars().all()
    records_by_id = {entry.id: row_to_record(entry) for entry in entries}
    expired = [
        entry
        for entry in entries
        if records_by_id[entry.id].is_expired()
        and not records_by_id[entry.id].retention.legal_hold
    ]
    for entry in expired:
        await session.delete(entry)
    await session.commit()
    return {"deleted": len(expired), "record_ids": [entry.id for entry in expired]}


# ---- Core Memory Blocks (path param routes last) ----


@router.get("/{block_label}", response_model=CoreMemoryBlockResponse)
async def get_core_memory_block(
    block_label: str,
    context: OrganizationContext = Depends(require_permission(Permission.MEMORY_READ)),
    session: AsyncSession = Depends(get_session),
):
    """View a specific CoreMemory block."""
    if block_label not in DEFAULT_LABELS:
        raise HTTPException(status_code=404, detail="Block not found")
    block = await _get_or_create_block(session, context, block_label)
    return CoreMemoryBlockResponse(
        label=block.label,
        value=block.value,
        limit=block.char_limit,
        read_only=False,
    )


@router.patch("/{block_label}", response_model=CoreMemoryBlockResponse)
async def update_core_memory_block(
    block_label: str,
    body: CoreMemoryBlockUpdate,
    context: OrganizationContext = Depends(require_permission(Permission.MEMORY_WRITE)),
    session: AsyncSession = Depends(get_session),
):
    """Manually edit a CoreMemory block."""
    if block_label not in DEFAULT_LABELS:
        raise HTTPException(status_code=404, detail="Block not found")

    block = await _get_or_create_block(session, context, block_label)

    if len(body.value) > block.char_limit:
        raise HTTPException(
            status_code=400,
            detail=f"Value exceeds block limit of {block.char_limit} characters",
        )

    block.value = body.value
    block.updated_at = datetime.now(timezone.utc)
    session.add(block)
    await session.commit()
    await session.refresh(block)

    return CoreMemoryBlockResponse(
        label=block.label,
        value=block.value,
        limit=block.char_limit,
        read_only=False,
    )
