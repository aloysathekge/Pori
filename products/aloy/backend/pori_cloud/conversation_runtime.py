"""Organization-scoped conversation memory loading and persistence."""

from __future__ import annotations

from datetime import datetime, timezone

from pori import AgentMemory
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from .memory_records import record_to_row, request_scope, row_to_record
from .models import (
    ContextArtifact,
    Conversation,
    CoreMemoryBlock,
    KnowledgeEntry,
    Message,
)


async def flush_context_artifact(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    conversation_id: str,
    run_id: str,
    memory: AgentMemory,
    diagnostics: dict | None,
) -> None:
    if not diagnostics or not diagnostics.get("dropped_messages"):
        return
    summary = next(
        (
            item
            for item in reversed(memory.summaries)
            if isinstance(item, dict) and item.get("summary")
        ),
        None,
    )
    if summary is None:
        return
    session.add(
        ContextArtifact(
            organization_id=organization_id,
            user_id=user_id,
            conversation_id=conversation_id,
            run_id=run_id,
            artifact_type="summary",
            content=str(summary["summary"]),
            source_message_ids=list(summary.get("source_message_ids") or []),
            diagnostics=diagnostics,
        )
    )


async def load_conversation_memory(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    conversation: Conversation,
) -> AgentMemory:
    agent_id = conversation.agent_config_id or "default_agent"
    memory = AgentMemory(
        organization_id=organization_id,
        user_id=user_id,
        agent_id=agent_id,
        session_id=conversation.id,
    )
    for label in ("persona", "human", "notes"):
        result = await session.execute(
            select(CoreMemoryBlock).where(
                CoreMemoryBlock.organization_id == organization_id,
                CoreMemoryBlock.user_id == user_id,
                CoreMemoryBlock.label == label,
            )
        )
        row = result.scalars().first()
        if row and row.value:
            memory.core_memory.get_block(label).set_value(row.value)

    result = await session.execute(
        select(KnowledgeEntry)
        .where(
            KnowledgeEntry.organization_id == organization_id,
            KnowledgeEntry.user_id == user_id,
        )
        .order_by(KnowledgeEntry.created_at.desc())
        .limit(100)
    )
    for entry in result.scalars().all():
        record = row_to_record(entry)
        if memory.scope.can_access(record.scope) and record.is_retrievable():
            record.metadata["legacy_collection"] = "experience"
            memory.memory_records.append(record)

    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at)
    )
    for message in result.scalars().all():
        memory.add_message(message.role, message.content)
    return memory


async def flush_conversation_memory(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    memory: AgentMemory,
) -> None:
    now = datetime.now(timezone.utc)
    for label in ("persona", "human", "notes"):
        block = memory.core_memory.get_block(label)
        result = await session.execute(
            select(CoreMemoryBlock).where(
                CoreMemoryBlock.organization_id == organization_id,
                CoreMemoryBlock.user_id == user_id,
                CoreMemoryBlock.label == label,
            )
        )
        existing = result.scalars().first()
        if existing:
            if existing.value != block.value:
                existing.value = block.value
                existing.updated_at = now
                session.add(existing)
        elif block.value:
            session.add(
                CoreMemoryBlock(
                    organization_id=organization_id,
                    user_id=user_id,
                    label=label,
                    value=block.value,
                )
            )

    scope = request_scope(
        user_id,
        organization_id=organization_id,
        agent_id=memory.agent_id,
        session_id=memory.session_id,
    )
    for record in memory.memory_records:
        if not scope.can_access(record.scope):
            continue
        existing = await session.get(KnowledgeEntry, record.id)
        session.add(record_to_row(record, existing))
