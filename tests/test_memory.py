import pytest

from pori.memory import AgentMemory, create_memory_store
from pori.memory_contracts import (
    ConflictPolicy,
    MemoryHit,
    MemoryKind,
    MemoryProvenance,
    MemoryRetention,
    MemoryScope,
    MemorySensitivity,
    MemoryStatus,
    evaluate_retrieval,
)


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


def test_memory_loads_legacy_namespace_and_writes_new_scope(tmp_path):
    db_path = tmp_path / "legacy-memory.db"
    store = create_memory_store(backend="sqlite", sqlite_path=str(db_path))
    store.save(
        "u1:a1:s1",
        {
            "meta": {},
            "experiences": [
                {
                    "id": "legacy-exp",
                    "text": "Legacy deployment note",
                    "importance": 2,
                    "meta": {"source": "user"},
                }
            ],
        },
    )

    memory = AgentMemory(
        organization_id="org-1",
        user_id="u1",
        agent_id="a1",
        session_id="s1",
        store=store,
    )

    assert memory.recall("deployment")[0][0] == "legacy-exp"
    memory.persist()
    assert store.load("org-1:u1:a1:s1") is not None


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


def test_memory_records_are_scoped_and_do_not_leak_between_tenants():
    memory = AgentMemory(
        organization_id="org-a",
        user_id="user-a",
        agent_id="agent-a",
        session_id="session-a",
    )
    own = memory.add_memory_record(
        "Customer prefers weekly reports",
        provenance=MemoryProvenance(source="user", source_id="message-1"),
    )
    memory.memory_records.append(
        own.model_copy(
            update={
                "id": "foreign-record",
                "scope": MemoryScope(
                    organization_id="org-b",
                    user_id="user-b",
                    agent_id="agent-a",
                    session_id="session-a",
                ),
                "content": "Foreign tenant secret",
            }
        )
    )

    hits = memory.search_memory_records("reports secret", k=10)

    assert [hit.record.id for hit in hits] == [own.id]
    evaluation = evaluate_retrieval(hits, [own.id], memory.scope)
    assert evaluation.recall_at_k == 1.0
    assert evaluation.leaked_record_ids == []


def test_memory_conflict_supersedes_previous_record():
    memory = AgentMemory(
        organization_id="org-a",
        user_id="user-a",
        agent_id="agent-a",
        session_id="session-a",
    )
    old = memory.add_memory_record(
        "The preferred database is SQLite",
        conflict_key="preferred-database",
    )
    new = memory.add_memory_record(
        "The preferred database is PostgreSQL",
        conflict_key="preferred-database",
        conflict_policy=ConflictPolicy.SUPERSEDE,
    )

    records = {record.id: record for record in memory.memory_records}
    assert records[old.id].status == MemoryStatus.SUPERSEDED
    assert records[old.id].superseded_by == new.id
    assert [hit.record.id for hit in memory.search_memory_records("database")] == [
        new.id
    ]


def test_memory_retention_delete_export_and_sensitivity():
    memory = AgentMemory(
        organization_id="org-a",
        user_id="user-a",
        agent_id="agent-a",
        session_id="session-a",
    )
    expired = memory.add_memory_record(
        "Temporary incident note",
        retention=MemoryRetention(delete_after="2000-01-01T00:00:00Z"),
    )
    restricted = memory.add_memory_record(
        "Payroll account details",
        sensitivity=MemorySensitivity.RESTRICTED,
        confidence=0.8,
    )

    assert memory.prune_expired_memory() == [expired.id]
    exported = memory.export_memory_records()
    assert [record["id"] for record in exported] == [restricted.id]

    assert memory.delete_memory_record(restricted.id) is True
    assert memory.search_memory_records("payroll") == []
    deleted_export = memory.export_memory_records(include_deleted=True)
    assert deleted_export[0]["status"] == "deleted"


def test_retrieval_evaluation_reports_precision_and_rank():
    scope = MemoryScope(
        organization_id="org-a",
        user_id="user-a",
        agent_id="agent-a",
        session_id="session-a",
    )
    memory = AgentMemory(
        organization_id=scope.organization_id,
        user_id=scope.user_id,
        agent_id=scope.agent_id or "agent-a",
        session_id=scope.session_id,
    )
    first = memory.add_memory_record(
        "Unrelated note",
        kind=MemoryKind.SEMANTIC,
    )
    expected = memory.add_memory_record(
        "Rollback checklist",
        kind=MemoryKind.PROCEDURAL,
    )
    hits = [
        MemoryHit(record=first, final_score=0.9),
        MemoryHit(record=expected, final_score=0.8),
    ]

    evaluation = evaluate_retrieval(hits, [expected.id], scope)

    assert evaluation.recall_at_k == 1.0
    assert evaluation.precision_at_k == 0.5
    assert evaluation.reciprocal_rank == 0.5
