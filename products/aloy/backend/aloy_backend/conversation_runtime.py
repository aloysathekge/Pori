"""Organization-scoped conversation memory loading and persistence."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from sqlalchemy import and_, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from pori import AgentMemory

from .config import settings
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
            if isinstance(item, dict)
            and item.get("summary")
            and item.get("source_start_message_id")
            and item.get("source_end_message_id")
            and item.get("source_message_count")
        ),
        None,
    )
    if summary is None:
        return
    # Serialize version allocation for this Conversation on databases that
    # support row locks. The partial unique index remains the final arbiter.
    await session.execute(
        select(Conversation.id)
        .where(Conversation.id == conversation_id)
        .with_for_update()
    )
    start_id = str(summary["source_start_message_id"])
    end_id = str(summary["source_end_message_id"])
    boundary_rows = list(
        (
            await session.execute(
                select(Message).where(col(Message.id).in_([start_id, end_id]))
            )
        )
        .scalars()
        .all()
    )
    boundary = {row.id: row for row in boundary_rows}
    start = boundary.get(start_id)
    end = boundary.get(end_id)
    if (
        start is None
        or end is None
        or start.conversation_id != conversation_id
        or end.conversation_id != conversation_id
        or (start.created_at, start.id) > (end.created_at, end.id)
    ):
        return
    first_message_id = (
        (
            await session.execute(
                select(Message.id)
                .where(
                    Message.conversation_id == conversation_id,
                    col(Message.role).in_(["user", "assistant"]),
                )
                .order_by(col(Message.created_at), col(Message.id))
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    covered_count = int(
        (
            await session.execute(
                select(func.count(col(Message.id))).where(
                    Message.conversation_id == conversation_id,
                    col(Message.role).in_(["user", "assistant"]),
                    or_(
                        col(Message.created_at) < end.created_at,
                        and_(
                            col(Message.created_at) == end.created_at,
                            col(Message.id) <= end.id,
                        ),
                    ),
                )
            )
        ).scalar_one()
    )
    if start_id != first_message_id or covered_count != int(
        summary["source_message_count"]
    ):
        # A summary may only advance a gap-free prefix. Very old legacy
        # Conversations that exceed the hydration ceiling fall back to the
        # bounded tail plus on-demand history search until explicitly backfilled.
        return

    content = str(summary["summary"])
    fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
    duplicate = (
        (
            await session.execute(
                select(ContextArtifact.id).where(
                    ContextArtifact.organization_id == organization_id,
                    ContextArtifact.conversation_id == conversation_id,
                    ContextArtifact.artifact_type == "summary",
                    ContextArtifact.content_fingerprint == fingerprint,
                    ContextArtifact.source_end_message_id == end_id,
                )
            )
        )
        .scalars()
        .first()
    )
    if duplicate is not None:
        return

    latest = (
        (
            await session.execute(
                select(ContextArtifact)
                .where(
                    ContextArtifact.organization_id == organization_id,
                    ContextArtifact.conversation_id == conversation_id,
                    ContextArtifact.artifact_type == "summary",
                    ContextArtifact.summary_version > 0,
                )
                .order_by(
                    col(ContextArtifact.summary_version).desc(),
                    col(ContextArtifact.created_at).desc(),
                )
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    if latest is not None and latest.source_ended_at is not None:
        if (end.created_at, end.id) <= (
            latest.source_ended_at,
            latest.source_end_message_id or "",
        ):
            return
    next_version = int(latest.summary_version if latest else 0) + 1
    session.add(
        ContextArtifact(
            organization_id=organization_id,
            user_id=user_id,
            event_id=event_id,
            conversation_id=conversation_id,
            run_id=run_id,
            artifact_type="summary",
            content=content,
            summary_version=next_version,
            source_start_message_id=start_id,
            source_end_message_id=end_id,
            source_started_at=start.created_at,
            source_ended_at=end.created_at,
            source_message_count=int(summary["source_message_count"]),
            content_fingerprint=fingerprint,
            # Keep this legacy export field bounded. Exact provenance is the
            # immutable ordered boundary + count, not an ever-growing JSON id list.
            source_message_ids=[start_id, end_id] if start_id != end_id else [start_id],
            diagnostics={
                **diagnostics,
                "summary_contract": "conversation-prefix-v1",
                "summary_version": next_version,
            },
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
    # Conversation rows and summary artifacts are host-owned. Never append
    # them to a possibly stale local memory-store snapshot.
    memory.reset_host_hydrated_context()
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
        # Canonical evidence/records are durable Event state, not mutable prompt
        # memory. Loading fetched excerpts into each turn would destroy context
        # budgets; loading records here would also let memory tools rewrite their
        # provenance envelope. Models read them through event_records_list, while
        # exact evidence remains inspectable through refs and the evidence endpoint.
        if (entry.metadata_ or {}).get("record_type") in {
            "web_evidence",
            "event_record",
            "research_report",
        }:
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

    latest_summary = None
    if conversation is not None:
        latest_summary = (
            (
                await session.execute(
                    select(ContextArtifact)
                    .where(
                        ContextArtifact.organization_id == organization_id,
                        ContextArtifact.user_id == user_id,
                        ContextArtifact.event_id == resolved_event_id,
                        ContextArtifact.conversation_id == current_session_id,
                        ContextArtifact.artifact_type == "summary",
                        ContextArtifact.summary_version > 0,
                        col(ContextArtifact.source_start_message_id).is_not(None),
                        col(ContextArtifact.source_end_message_id).is_not(None),
                    )
                    .order_by(
                        col(ContextArtifact.summary_version).desc(),
                        col(ContextArtifact.created_at).desc(),
                    )
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )

    boundary_message = None
    if latest_summary is not None and latest_summary.source_end_message_id:
        candidate = await session.get(Message, latest_summary.source_end_message_id)
        expected_fingerprint = hashlib.sha256(
            latest_summary.content.encode("utf-8")
        ).hexdigest()
        if (
            candidate is not None
            and candidate.conversation_id == current_session_id
            and latest_summary.content_fingerprint == expected_fingerprint
        ):
            boundary_message = candidate
            memory.hydrate_context_summary(
                latest_summary.content,
                version=latest_summary.summary_version,
                source_start_message_id=str(latest_summary.source_start_message_id),
                source_end_message_id=str(latest_summary.source_end_message_id),
                source_message_count=latest_summary.source_message_count,
                source_started_at=latest_summary.source_started_at,
                source_ended_at=latest_summary.source_ended_at,
                content_fingerprint=latest_summary.content_fingerprint,
            )

    # Hydrate only the current Conversation tail. Sibling and older Event
    # messages page fault through the async search tool instead of making every
    # Run load and embed thousands of rows.
    statement = select(Message).where(Message.conversation_id == current_session_id)
    if boundary_message is not None:
        statement = statement.where(
            or_(
                col(Message.created_at) > boundary_message.created_at,
                and_(
                    col(Message.created_at) == boundary_message.created_at,
                    col(Message.id) > boundary_message.id,
                ),
            )
        )
    if exclude_message_id:
        statement = statement.where(Message.id != exclude_message_id)
    rows = list(
        reversed(
            (
                (
                    await session.execute(
                        statement.order_by(
                            col(Message.created_at).desc(), col(Message.id).desc()
                        ).limit(settings.conversation_hydration_max_messages)
                    )
                )
                .scalars()
                .all()
            )
        )
    )
    rendered_rows: list[tuple[Message, str]] = []
    for message in rows:
        body = message.content
        for file in (message.metadata_ or {}).get("files", []) or []:
            if file.get("content"):
                body += f"\n\n<file name=\"{file.get('name', 'file')}\">\n{file['content']}\n</file>"
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

    selected: list[tuple[Message, str]] = []
    remaining_chars = settings.conversation_hydration_max_chars
    for message, body in reversed(rendered_rows):
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
