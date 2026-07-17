from __future__ import annotations

import pytest

from aloy_backend.conversation_runtime import load_event_memory
from aloy_backend.models import (
    Conversation,
    Event,
    KnowledgeEntry,
    Message,
    StoredFile,
    Task,
)


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
    history = memory.search_event_history("sibling A message")
    assert any(
        hit["content"] == "[Session session-a2] sibling A message" for hit in history
    )


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
