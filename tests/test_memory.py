import pytest

from pori.memory import AgentMemory, create_memory_store


def test_agent_memory_basic_flow(legacy_memory, test_task_id):
    """Basic create/read/update flow on AgentMemory should work."""
    memory = legacy_memory

    # Create a task and verify it's tracked
    task = memory.create_task(test_task_id, "Demo task")
    assert task.task_id == test_task_id
    assert task.status == "in_progress"

    # Add a message and verify it appears in history
    memory.add_message("user", "Hello")
    assert len(memory.messages) == 1
    assert memory.messages[0].content == "Hello"

    # Update and read simple state
    memory.update_state("foo", "bar")
    assert memory.get_state("foo") == "bar"

    # Add a tool call record
    memory.add_tool_call(
        tool_name="echo",
        parameters={"text": "hi"},
        result={"ok": True},
        success=True,
    )
    assert len(memory.tool_call_history) == 1
    assert memory.tool_call_history[0].tool_name == "echo"

    # Create a summary string and check it contains expected pieces
    summary = memory.create_summary(step=1)
    assert summary.startswith("Step 1 summary:")
    assert "messages" in summary
    assert "tool calls" in summary

    # Final answer is not set by default
    assert memory.get_final_answer() is None


def test_conversation_search(legacy_memory):
    memory = legacy_memory
    memory.add_message("user", "Hello world")
    memory.add_message("assistant", "General Aloy")
    memory.add_message("user", "Hello again")

    results = memory.conversation_search(query="hello", limit=10)
    assert len(results) == 2
    assert results[0]["content"] == "Hello again"
    assert results[1]["content"] == "Hello world"

    filtered = memory.conversation_search(query="hello", limit=10, roles=["assistant"])
    assert filtered == []


def test_archival_memory_insert_and_search(legacy_memory):
    memory = legacy_memory
    pid = memory.archival_memory_insert(
        text="Project spec: use Postgres and FastAPI",
        tags=["spec", "backend"],
        importance=3,
    )
    assert pid.startswith("arch_")

    hits = memory.archival_memory_search(query="FastAPI", k=5, min_score=0.0)
    assert len(hits) >= 1
    assert hits[0][0] == pid
    assert "FastAPI" in hits[0][1]
    assert isinstance(memory.archival_passages[0].get("embedding"), list)
    assert len(memory.archival_passages[0]["embedding"]) > 0


def test_memory_persistence_with_sqlite_store(tmp_path):
    db_path = tmp_path / "memory.db"
    store = create_memory_store(backend="sqlite", sqlite_path=str(db_path))

    mem1 = AgentMemory(
        user_id="u1",
        agent_id="a1",
        session_id="s1",
        store=store,
    )
    mem1.add_message("user", "Persist me")
    mem1.add_experience("Known fact", importance=2)
    mem1.archival_memory_insert("Long-term note", tags=["note"], importance=3)
    mem1.core_memory.get_block("notes").append("Persistent core note")
    mem1.persist()

    mem2 = AgentMemory(
        user_id="u1",
        agent_id="a1",
        session_id="s1",
        store=store,
    )
    assert any(m.content == "Persist me" for m in mem2.messages)
    assert any("Known fact" in str(e.get("text", "")) for e in mem2.experiences)
    assert any(
        "Long-term note" in str(p.get("text", "")) for p in mem2.archival_passages
    )
    assert "Persistent core note" in mem2.core_memory.get_block("notes").value


def test_memory_serializable_state_round_trip(legacy_memory):
    memory = legacy_memory
    memory.add_message("user", "hello")
    state = memory.export_state()

    rebuilt = AgentMemory.from_state(state, store=memory.store)
    assert rebuilt.namespace == memory.namespace
    assert rebuilt.export_state().session_id == state.session_id


def test_token_limited_messages_generates_summary(legacy_memory):
    memory = legacy_memory
    for i in range(40):
        memory.add_message("user", f"Message number {i} " + ("x" * 120))

    window = memory.get_token_limited_messages(
        max_tokens=300,
        reserve_tokens=200,
        include_summary_message=True,
    )
    assert len(window) > 0
    assert window[0]["role"] == "system"
    assert "Conversation summary" in window[0]["content"]
    assert len(memory.summaries) >= 1


def test_token_limited_messages_reuses_cached_summary(legacy_memory):
    memory = legacy_memory
    for i in range(30):
        memory.add_message("user", f"Cached summary message {i} " + ("z" * 80))

    first = memory.get_token_limited_messages(
        max_tokens=280,
        reserve_tokens=180,
        include_summary_message=True,
    )
    summary_count_after_first = len(memory.summaries)
    second = memory.get_token_limited_messages(
        max_tokens=280,
        reserve_tokens=180,
        include_summary_message=True,
    )

    assert first[0]["role"] == "system"
    assert second[0]["role"] == "system"
    assert first[0]["content"] == second[0]["content"]
    assert len(memory.summaries) == summary_count_after_first


def test_recall_uses_embedded_experiences(legacy_memory):
    memory = legacy_memory
    key = memory.add_experience(
        "Deployment runbook includes rollback checklist for production incidents",
        importance=3,
    )

    hits = memory.recall("rollback checklist", k=3, min_score=0.0)
    assert len(hits) >= 1
    assert hits[0][0] == key
    assert isinstance(memory.experiences[0].get("embedding"), list)


def test_in_memory_store_instances_are_isolated():
    """Regression: InMemoryMemoryStore used a class-level dict, so two
    instances leaked state into each other. Each instance must have its own
    namespace-keyed store."""
    from pori.memory import InMemoryMemoryStore

    store_a = InMemoryMemoryStore()
    store_b = InMemoryMemoryStore()

    store_a.save("ns", {"owner": "a"})
    assert store_b.load("ns") is None
    assert store_a.load("ns") == {"owner": "a"}

    store_b.save("ns", {"owner": "b"})
    assert store_a.load("ns") == {"owner": "a"}
    assert store_b.load("ns") == {"owner": "b"}


def test_hydrate_timestamps_parses_iso_strings_in_place():
    """Timestamp hydration helper should convert ISO strings to datetime on each
    dict item and leave non-dict or missing timestamps alone."""
    from datetime import datetime

    from pori.memory import AgentMemory

    memory = AgentMemory()
    items = [
        {"timestamp": "2024-01-02T03:04:05"},
        {"timestamp": datetime(2024, 1, 1)},
        {"no_timestamp": True},
        "not a dict",
    ]
    memory._hydrate_timestamps(items)

    assert items[0]["timestamp"] == datetime(2024, 1, 2, 3, 4, 5)
    assert items[1]["timestamp"] == datetime(2024, 1, 1)
    assert "timestamp" not in items[2]
    assert items[3] == "not a dict"
