from pori import (
    Agent,
    AgentMemory,
    DefaultContextEngine,
    RetrievalEvidence,
    SessionMessage,
    SessionRecord,
    SQLiteSessionRepository,
    fuse_retrieval,
)


def test_context_engine_reports_compaction_and_preserves_recent_tail():
    memory = AgentMemory()
    for index in range(20):
        memory.add_message("user", f"message {index} " + ("x" * 100))

    window = DefaultContextEngine().build(
        memory,
        max_tokens=300,
        reserve_tokens=0,
    )

    assert window.diagnostics.dropped_messages > 0
    assert window.diagnostics.summary_included is True
    assert window.diagnostics.reason == "compacted"
    assert window.messages[-1]["content"].startswith("message 19")


def test_agent_freezes_core_and_retrieved_memory_for_one_run(mock_llm, tool_registry):
    memory = AgentMemory()
    memory.core_memory.get_block("human").set_value("Original preference")
    original_id = memory.add_experience("release checklist alpha", importance=5)
    agent = Agent(
        task="release checklist alpha",
        llm=mock_llm,
        tools_registry=tool_registry,
        memory=memory,
    )

    memory.core_memory.get_block("human").set_value("Changed mid-run")
    memory.add_experience("release checklist beta", importance=5)
    messages = agent._build_messages()
    rendered = "\n".join(message.content for message in messages)

    assert "Original preference" in messages[0].content
    assert "Changed mid-run" not in messages[0].content
    assert original_id in rendered
    assert "release checklist beta" not in rendered


def test_sqlite_session_repository_lifecycle_and_scope(tmp_path):
    repository = SQLiteSessionRepository(tmp_path / "sessions.db")
    parent = repository.create(
        SessionRecord(
            organization_id="org-a",
            user_id="alice",
            title="Investigation",
        )
    )
    first = repository.add_message(
        SessionMessage(
            session_id=parent.id,
            role="user",
            content="PostgreSQL migration checklist",
        )
    )
    repository.add_message(
        SessionMessage(
            session_id=parent.id,
            role="assistant",
            content="Validate the migration head",
        )
    )

    assert repository.search("org-a", "alice", "PostgreSQL migration")
    assert repository.search("org-a", "mallory", "PostgreSQL migration") == []

    child = repository.branch(
        "org-a",
        "alice",
        parent.id,
        through_message_id=first.id,
        title="Alternative",
    )
    exported = repository.export("org-a", "alice", child.id)
    assert child.parent_session_id == parent.id
    assert child.branched_from_message_id == first.id
    assert len(exported.messages) == 1
    assert exported.messages[0].metadata["copied_from_message_id"] == first.id
    assert repository.delete("org-a", "alice", child.id) is True
    assert repository.get("org-a", "alice", child.id) is None


def test_retrieval_fusion_preserves_source_identity():
    fused = fuse_retrieval(
        [
            RetrievalEvidence(
                source_type="session",
                source_id="message-1",
                session_id="session-1",
                content="A session fact",
                score=0.8,
                provenance={"role": "user"},
            )
        ],
        [
            RetrievalEvidence(
                source_type="memory",
                source_id="memory-1",
                content="A durable fact",
                score=0.9,
                provenance={"source": "user"},
            )
        ],
    )

    assert [item.source_type for item in fused] == ["memory", "session"]
    assert fused[0].provenance == {"source": "user"}
