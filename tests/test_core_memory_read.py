"""Tests for core_memory_read tool and prompt guidance."""

import pytest

from pori.memory import AgentMemory
from pori.tools.registry import ToolExecutor, tool_registry
from pori.tools.standard import register_all_tools
from pori.utils.prompt_loader import load_prompt


@pytest.fixture
def memory_with_core_content():
    """Create memory with populated core memory blocks."""
    memory = AgentMemory()
    memory.core_memory.get_block("persona").append("I am a helpful assistant.")
    memory.core_memory.get_block("human").append("User prefers short answers.")
    memory.core_memory.get_block("notes").append("Meeting at 3pm today.")
    return memory


@pytest.fixture
def executor_with_core_tools():
    """Create a tool executor with all standard tools registered."""
    registry = tool_registry()
    register_all_tools(registry)
    return ToolExecutor(registry)


class TestCoreMemoryReadTool:
    """Tests for the core_memory_read tool."""

    def test_read_single_block(
        self, executor_with_core_tools, memory_with_core_content
    ):
        """core_memory_read should return a single block when label is provided."""
        ctx = {"memory": memory_with_core_content, "state": {}}
        result = executor_with_core_tools.execute_tool(
            "core_memory_read",
            {"label": "persona"},
            context=ctx,
        )
        assert result["success"] is True
        inner = result.get("result", result)
        assert "blocks" in inner
        assert "persona" in inner["blocks"]
        assert "helpful assistant" in inner["blocks"]["persona"]
        assert "human" not in inner["blocks"]
        assert "notes" not in inner["blocks"]

    def test_read_all_blocks(self, executor_with_core_tools, memory_with_core_content):
        """core_memory_read should return all blocks when no label is provided."""
        ctx = {"memory": memory_with_core_content, "state": {}}
        result = executor_with_core_tools.execute_tool(
            "core_memory_read",
            {},
            context=ctx,
        )
        assert result["success"] is True
        inner = result.get("result", result)
        assert "blocks" in inner
        assert "persona" in inner["blocks"]
        assert "human" in inner["blocks"]
        assert "notes" in inner["blocks"]
        assert "helpful assistant" in inner["blocks"]["persona"]
        assert "short answers" in inner["blocks"]["human"]
        assert "Meeting" in inner["blocks"]["notes"]

    def test_read_does_not_mutate(
        self, executor_with_core_tools, memory_with_core_content
    ):
        """core_memory_read should not modify any memory blocks."""
        ctx = {"memory": memory_with_core_content, "state": {}}
        original_persona = memory_with_core_content.core_memory.get_block(
            "persona"
        ).value
        original_human = memory_with_core_content.core_memory.get_block("human").value
        original_notes = memory_with_core_content.core_memory.get_block("notes").value

        executor_with_core_tools.execute_tool(
            "core_memory_read",
            {},
            context=ctx,
        )

        assert (
            memory_with_core_content.core_memory.get_block("persona").value
            == original_persona
        )
        assert (
            memory_with_core_content.core_memory.get_block("human").value
            == original_human
        )
        assert (
            memory_with_core_content.core_memory.get_block("notes").value
            == original_notes
        )

    def test_read_nonexistent_block_creates_empty(
        self, executor_with_core_tools, memory_with_core_content
    ):
        """core_memory_read returns an empty block for unknown labels (matches CoreMemory behavior)."""
        ctx = {"memory": memory_with_core_content, "state": {}}
        result = executor_with_core_tools.execute_tool(
            "core_memory_read",
            {"label": "unknown_block"},
            context=ctx,
        )
        # CoreMemory.get_block creates a new empty block for unknown labels
        inner = result.get("result", result)
        assert inner.get("success") is True
        assert inner["blocks"]["unknown_block"] == ""


class TestPromptGuidance:
    """Tests for prompt guidance on memory tools."""

    def test_system_prompt_references_core_memory_read(self):
        """System prompt should mention core_memory_read for inspection."""
        prompt = load_prompt("system/agent_core.md")
        assert "core_memory_read" in prompt
        assert "inspect" in prompt.lower() or "read" in prompt.lower()

    def test_system_prompt_warns_against_rethink_for_inspection(self):
        """System prompt should warn against using rethink tools for inspection."""
        prompt = load_prompt("system/agent_core.md")
        assert "memory_rethink" in prompt or "core_memory_rethink" in prompt
        assert "not" in prompt.lower() or "never" in prompt.lower()
