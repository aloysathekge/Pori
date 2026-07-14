from pathlib import Path

from pori import AgentMemory, get_workspace_data
from pori.memory_contracts import MemoryScope


def test_event_workspace_is_shared_but_run_scratch_is_isolated(tmp_path):
    first = get_workspace_data("event-a", "run-1", str(tmp_path))
    second = get_workspace_data("event-a", "run-2", str(tmp_path))
    other = get_workspace_data("event-b", "run-1", str(tmp_path))

    assert first.workspace_path == second.workspace_path
    assert first.uploads_path == second.uploads_path
    assert first.outputs_path != second.outputs_path
    assert Path(first.outputs_path).parts[-3:] == ("runs", "run-1", "scratch")
    assert first.workspace_path != other.workspace_path


def test_event_memory_scope_accepts_global_and_same_event_only():
    request = MemoryScope(organization_id="org-1", user_id="alice", event_id="event-a")

    assert request.can_access(MemoryScope(organization_id="org-1", user_id="alice"))
    assert request.can_access(
        MemoryScope(organization_id="org-1", user_id="alice", event_id="event-a")
    )
    assert request.can_access(
        MemoryScope(
            organization_id="org-1",
            user_id="alice",
            event_id="event-a",
            session_id="another-session",
        )
    )
    assert not request.can_access(
        MemoryScope(organization_id="org-1", user_id="alice", event_id="event-b")
    )


def test_event_history_search_uses_corpus_outside_prompt_window():
    memory = AgentMemory(event_id="event-a")
    memory.add_message("user", "recent prompt message")
    memory.index_event_history(
        [
            {"role": "user", "content": "old searchable launch codename kestrel"},
            {"role": "assistant", "content": "recent prompt message"},
        ]
    )

    results = memory.search_event_history("kestrel")

    assert results[0]["content"] == "old searchable launch codename kestrel"
