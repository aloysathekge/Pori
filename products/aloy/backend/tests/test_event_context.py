from __future__ import annotations

import hashlib
import importlib
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect
from sqlmodel import select

from aloy_backend.conversation_runtime import (
    flush_context_artifact,
    load_event_memory,
)
from aloy_backend.models import (
    ContextArtifact,
    Conversation,
    Event,
    KnowledgeEntry,
    Message,
    StoredFile,
    Task,
)
from aloy_backend.tools.event_history import EventHistorySearchHandler


async def test_event_memory_loads_global_and_same_event_without_leakage(
    db_session_maker,
):
    async with db_session_maker() as session:
        event_a = Event(id="evt-a", organization_id="org-1", user_id="alice", title="A")
        event_b = Event(id="evt-b", organization_id="org-1", user_id="alice", title="B")
        session.add_all([event_a, event_b])
        session_a1 = Conversation(
            id="session-a1",
            organization_id="org-1",
            user_id="alice",
            event_id=event_a.id,
        )
        session_a2 = Conversation(
            id="session-a2",
            organization_id="org-1",
            user_id="alice",
            event_id=event_a.id,
        )
        session_b = Conversation(
            id="session-b",
            organization_id="org-1",
            user_id="alice",
            event_id=event_b.id,
        )
        session.add_all([session_a1, session_a2, session_b])
        session.add_all(
            [
                KnowledgeEntry(
                    id="global-memory",
                    organization_id="org-1",
                    user_id="alice",
                    content="global preference",
                ),
                KnowledgeEntry(
                    id="event-a-memory",
                    organization_id="org-1",
                    user_id="alice",
                    event_id=event_a.id,
                    content="A knowledge",
                ),
                KnowledgeEntry(
                    id="event-b-memory",
                    organization_id="org-1",
                    user_id="alice",
                    event_id=event_b.id,
                    content="B secret knowledge",
                ),
                Message(
                    conversation_id=session_a1.id,
                    role="user",
                    content="current A message",
                ),
                Message(
                    conversation_id=session_a2.id,
                    role="assistant",
                    content="sibling A message",
                ),
                Message(
                    conversation_id=session_b.id,
                    role="user",
                    content="B secret message",
                ),
                Task(
                    organization_id="org-1",
                    user_id="alice",
                    event_id=event_a.id,
                    title="A task",
                    created_by="alice",
                ),
                Task(
                    organization_id="org-1",
                    user_id="alice",
                    event_id=event_b.id,
                    title="B secret task",
                    created_by="alice",
                ),
                StoredFile(
                    id="file-a",
                    organization_id="org-1",
                    user_id="alice",
                    event_id=event_a.id,
                    name="a.txt",
                    storage_key="unused/a",
                ),
                StoredFile(
                    id="file-b",
                    organization_id="org-1",
                    user_id="alice",
                    event_id=event_b.id,
                    name="secret-b.txt",
                    storage_key="unused/b",
                ),
            ]
        )
        await session.commit()

        memory = await load_event_memory(
            session,
            organization_id="org-1",
            user_id="alice",
            conversation=session_a1,
        )

    knowledge = {record.content for record in memory.memory_records}
    assert knowledge == {"global preference", "A knowledge"}
    rendered = "\n".join(message.content for message in memory.messages)
    assert "current A message" in rendered
    assert "sibling A message" not in rendered
    assert memory.trusted_context is not None
    assert "A task" in memory.trusted_context
    assert "a.txt" in memory.trusted_context
    assert "B secret" not in rendered
    assert "secret-b.txt" not in rendered
    assert "B secret" not in memory.trusted_context
    assert "secret-b.txt" not in memory.trusted_context
    assert memory.trusted_context_fingerprint
    # Sibling history is no longer eagerly loaded into every Run.
    eager_history = memory.search_event_history("sibling A message")
    assert all(hit["content"] != "sibling A message" for hit in eager_history)
    handler = EventHistorySearchHandler(
        run_context=SimpleNamespace(
            organization_id="org-1", user_id="alice", event_id="evt-a"
        ),
        session_factory=db_session_maker,
    )
    history = await handler.search(query="sibling A message", limit=10)
    assert any(hit["content"] == "sibling A message" for hit in history)
    assert all("B secret" not in hit["content"] for hit in history)


async def test_event_loader_rejects_cross_tenant_event(db_session_maker):
    async with db_session_maker() as session:
        event = Event(
            id="evt-private",
            organization_id="org-private",
            user_id="bob",
            title="Private",
        )
        session.add(event)
        await session.commit()

        with pytest.raises(ValueError, match="unavailable"):
            await load_event_memory(
                session,
                organization_id="org-1",
                user_id="alice",
                event_id=event.id,
            )


async def test_versioned_summary_hydrates_only_the_unsummarized_tail(
    db_session_maker,
):
    async with db_session_maker() as session:
        event = Event(
            id="evt-long", organization_id="org-1", user_id="alice", title="Long"
        )
        conversation = Conversation(
            id="session-long",
            organization_id="org-1",
            user_id="alice",
            event_id=event.id,
        )
        first = Message(
            id="message-001",
            conversation_id=conversation.id,
            role="user",
            content="ancient request",
        )
        second = Message(
            id="message-002",
            conversation_id=conversation.id,
            role="assistant",
            content="ancient decision",
        )
        tail = Message(
            id="message-003",
            conversation_id=conversation.id,
            role="user",
            content="recent follow-up",
        )
        session.add_all([event, conversation, first, second, tail])
        await session.flush()
        content = "Verified summary of the ancient request and decision."
        session.add(
            ContextArtifact(
                organization_id="org-1",
                user_id="alice",
                event_id=event.id,
                conversation_id=conversation.id,
                artifact_type="summary",
                content=content,
                summary_version=1,
                source_start_message_id=first.id,
                source_end_message_id=second.id,
                source_started_at=first.created_at,
                source_ended_at=second.created_at,
                source_message_count=2,
                content_fingerprint=hashlib.sha256(content.encode()).hexdigest(),
                source_message_ids=[first.id, second.id],
            )
        )
        await session.commit()

        memory = await load_event_memory(
            session,
            organization_id="org-1",
            user_id="alice",
            conversation=conversation,
        )

    assert [message.id for message in memory.messages] == [tail.id]
    window = memory.get_token_limited_messages(max_tokens=3_000, reserve_tokens=500)
    assert window[0] == {"role": "system", "content": content}
    assert window[1]["content"] == "recent follow-up"


async def test_fresh_life_conversation_is_clean_but_history_pages_on_demand(
    db_session_maker,
):
    async with db_session_maker() as session:
        life = Event(
            id="evt-life",
            organization_id="org-1",
            user_id="alice",
            title="Life",
            is_life=True,
        )
        old = Conversation(
            id="life-old",
            organization_id="org-1",
            user_id="alice",
            event_id=life.id,
        )
        fresh = Conversation(
            id="life-fresh",
            organization_id="org-1",
            user_id="alice",
            event_id=life.id,
        )
        session.add_all(
            [
                life,
                old,
                fresh,
                Message(
                    conversation_id=old.id,
                    role="user",
                    content="my older private transcript detail",
                ),
                KnowledgeEntry(
                    id="accepted-personal-memory",
                    organization_id="org-1",
                    user_id="alice",
                    content="Alice prefers concise answers",
                ),
            ]
        )
        await session.commit()
        memory = await load_event_memory(
            session,
            organization_id="org-1",
            user_id="alice",
            conversation=fresh,
        )

    assert memory.messages == []
    assert {record.content for record in memory.memory_records} == {
        "Alice prefers concise answers"
    }
    handler = EventHistorySearchHandler(
        run_context=SimpleNamespace(
            organization_id="org-1", user_id="alice", event_id=life.id
        ),
        session_factory=db_session_maker,
    )
    hits = await handler.search(query="older private transcript", limit=5)
    assert [hit["conversation_id"] for hit in hits] == [old.id]


async def test_summary_flush_is_versioned_bounded_and_idempotent(db_session_maker):
    async with db_session_maker() as session:
        event = Event(
            id="evt-flush", organization_id="org-1", user_id="alice", title="Flush"
        )
        conversation = Conversation(
            id="session-flush",
            organization_id="org-1",
            user_id="alice",
            event_id=event.id,
        )
        first = Message(
            id="flush-message-1",
            conversation_id=conversation.id,
            role="user",
            content="large old context",
        )
        session.add_all([event, conversation, first])
        await session.commit()
        memory = await load_event_memory(
            session,
            organization_id="org-1",
            user_id="alice",
            conversation=conversation,
        )
        memory.store_context_summary(
            [first.id],
            "A faithful compacted summary.",
            provenance={
                "source_start_message_id": first.id,
                "source_end_message_id": first.id,
                "source_message_count": 1,
                "source_started_at": first.created_at.isoformat(),
                "source_ended_at": first.created_at.isoformat(),
            },
        )
        diagnostics = {"dropped_messages": 1, "reason": "compacted"}
        kwargs = {
            "organization_id": "org-1",
            "user_id": "alice",
            "event_id": event.id,
            "conversation_id": conversation.id,
            "run_id": "run-summary-1",
            "memory": memory,
            "diagnostics": diagnostics,
        }
        await flush_context_artifact(session, **kwargs)
        await session.flush()
        await flush_context_artifact(session, **kwargs)
        await session.flush()

        artifacts = list(
            (
                await session.execute(
                    select(ContextArtifact).where(
                        ContextArtifact.conversation_id == conversation.id
                    )
                )
            )
            .scalars()
            .all()
        )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.summary_version == 1
    assert artifact.source_start_message_id == first.id
    assert artifact.source_end_message_id == first.id
    assert artifact.source_message_count == 1
    assert artifact.source_message_ids == [first.id]
    assert artifact.diagnostics["summary_contract"] == "conversation-prefix-v1"


def test_context_summary_boundary_migration_round_trip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'context-summary.db'}")
    metadata = sa.MetaData()
    sa.Table(
        "context_artifacts",
        metadata,
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("conversation_id", sa.String(), nullable=False),
        sa.Column("artifact_type", sa.String(), nullable=False),
    )
    metadata.create_all(engine)
    expected = {
        "summary_version",
        "source_start_message_id",
        "source_end_message_id",
        "source_started_at",
        "source_ended_at",
        "source_message_count",
        "content_fingerprint",
    }

    with engine.begin() as connection:
        migration = importlib.import_module(
            "aloy_backend.alembic.versions." "l4c5d6e7f8a9_context_summary_boundaries"
        )
        original_op = migration.op
        migration.op = Operations(MigrationContext.configure(connection))
        try:
            migration.upgrade()
            columns = {
                column["name"]
                for column in inspect(connection).get_columns("context_artifacts")
            }
            assert expected <= columns
            migration.downgrade()
            columns = {
                column["name"]
                for column in inspect(connection).get_columns("context_artifacts")
            }
            assert columns == {"id", "conversation_id", "artifact_type"}
        finally:
            migration.op = original_op
    engine.dispose()
