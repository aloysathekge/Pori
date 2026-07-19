"""Organization-scoped conversation memory loading and persistence."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from pori import AgentMemory

from .event_context import refresh_event_context_snapshot, render_event_context_pack
from .memory_records import record_to_row, request_scope, row_to_record
from .models import (
    ContextArtifact,
    Conversation,
    CoreMemoryBlock,
    Event,
    KnowledgeEntry,
    Message,
)
from .provisioning import migrate_legacy_session_workspace
from .scope_resolver import ORG, resolve_layered


async def flush_context_artifact(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    event_id: str,
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
            event_id=event_id,
            conversation_id=conversation_id,
            run_id=run_id,
            artifact_type="summary",
            content=str(summary["summary"]),
            source_message_ids=list(summary.get("source_message_ids") or []),
            diagnostics=diagnostics,
        )
    )


async def load_event_memory(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    conversation: Conversation | None = None,
    event_id: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    exclude_message_id: str | None = None,
) -> AgentMemory:
    resolved_event_id = event_id or (conversation.event_id if conversation else None)
    if not resolved_event_id:
        raise ValueError("Event identity is required to load memory")
    event = await session.get(Event, resolved_event_id)
    if (
        event is None
        or event.organization_id != organization_id
        or event.user_id != user_id
    ):
        raise ValueError("Event is unavailable")
    current_session_id = session_id or (
        conversation.id if conversation else resolved_event_id
    )
    if conversation is not None:
        migrate_legacy_session_workspace(conversation.id, resolved_event_id)
    resolved_agent_id = (
        agent_id
        or (conversation.agent_config_id if conversation else None)
        or "default_agent"
    )
    memory = AgentMemory(
        organization_id=organization_id,
        user_id=user_id,
        event_id=resolved_event_id,
        agent_id=resolved_agent_id,
        session_id=current_session_id,
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

    # The moat: assemble knowledge from the org + personal layers (team slots in
    # once membership exists), then let the most-specific level win per
    # ``conflict_key`` (personal > team > org). See scope_resolver.py.
    entries_result = await session.execute(
        select(KnowledgeEntry)
        .where(
            KnowledgeEntry.organization_id == organization_id,
            or_(
                col(KnowledgeEntry.user_id) == user_id,
                col(KnowledgeEntry.scope_level) == ORG,
            ),
            or_(
                col(KnowledgeEntry.event_id).is_(None),
                col(KnowledgeEntry.event_id) == resolved_event_id,
            ),
        )
        .order_by(col(KnowledgeEntry.created_at).desc())
        .limit(200)
    )
    for entry in resolve_layered(
        list(entries_result.scalars().all()), event_id=resolved_event_id
    ):
        # Old file-library pointers predate Event memory and may have no
        # event_id. Never let those legacy rows reveal a file name in an
        # unrelated Event; current pointers are written with their owner.
        if "file-library" in (entry.tags or []) and entry.event_id != resolved_event_id:
            continue
        record = row_to_record(entry)
        if not record.is_retrievable():
            continue
        # Org-level knowledge is org-shared, so it bypasses the per-user access
        # check; team/personal stay user-scoped.
        if entry.scope_level != ORG and not memory.scope.can_access(record.scope):
            continue
        record.metadata["legacy_collection"] = "experience"
        memory.memory_records.append(record)

    snapshot, _pack, _created = await refresh_event_context_snapshot(
        session,
        organization_id=organization_id,
        user_id=user_id,
        event_id=resolved_event_id,
    )
    memory.set_trusted_context(
        render_event_context_pack(snapshot),
        fingerprint=snapshot.fingerprint,
        cacheable=snapshot.provider_cache_allowed,
    )

    statement = (
        select(Message, Conversation.id)
        .join(Conversation, col(Conversation.id) == col(Message.conversation_id))
        .where(
            Conversation.organization_id == organization_id,
            Conversation.user_id == user_id,
            Conversation.event_id == resolved_event_id,
        )
        .order_by(col(Message.created_at).desc(), col(Message.id).desc())
        .limit(5000)
    )
    if exclude_message_id:
        statement = statement.where(Message.id != exclude_message_id)
    rows = list(reversed((await session.execute(statement)).all()))
    rendered_rows: list[tuple[Message, str]] = []
    for message, row_session_id in rows:
        body = message.content
        for file in (message.metadata_ or {}).get("files", []) or []:
            if file.get("content"):
                body += f"\n\n<file name=\"{file.get('name', 'file')}\">\n{file['content']}\n</file>"
        if row_session_id != current_session_id:
            body = f"[Session {row_session_id}] {body}"
        rendered_rows.append((message, body))
    memory.index_event_history(
        [
            {
                "id": message.id,
                "role": message.role,
                "content": body,
                "timestamp": message.created_at,
            }
            for message, body in rendered_rows
        ]
    )

    # Sibling Conversations remain searchable as scoped Event history, but a
    # fresh Life thread must not receive their transcript automatically.
    current_rows = [
        (message, body)
        for message, body in rendered_rows
        if message.conversation_id == current_session_id
    ]
    selected: list[tuple[Message, str]] = []
    remaining_chars = 100_000
    for message, body in reversed(current_rows):
        if len(body) > remaining_chars and selected:
            break
        selected.append((message, body[-remaining_chars:]))
        remaining_chars -= min(len(body), remaining_chars)
        if remaining_chars <= 0:
            break
    memory.hydrate_messages(
        [
            {
                "id": message.id,
                "role": message.role,
                "content": body,
                "timestamp": message.created_at,
            }
            for message, body in reversed(selected)
        ]
    )
    return memory


async def flush_event_memory(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    event_id: str,
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
        event_id=event_id,
        agent_id=memory.agent_id,
        session_id=memory.session_id,
    )
    for record in memory.memory_records:
        if not scope.can_access(record.scope):
            continue
        existing_entry = await session.get(KnowledgeEntry, record.id)
        session.add(record_to_row(record, existing_entry))


# Compatibility aliases for product code migrating in stacked phases.
load_conversation_memory = load_event_memory
flush_conversation_memory = flush_event_memory
