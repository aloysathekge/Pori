import pytest


def test_agent_memory_basic_flow(legacy_memory, test_task_id):
    """Basic create/read/update flow on AgentMemory should work."""
    memory = legacy_memory

    # Create a task and verify it's tracked
    task = memory.create_task(test_task_id, "Demo task")
    assert task.task_id == test_task_id
    assert task.status == "in_progress"

    # Add a message and verify it appears in history
    memory.add_message("user", "Hello")
    assert len(memory.conversation_history) == 1
    assert memory.conversation_history[0].content == "Hello"

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
