"""Tests for Evaluator task completion logic."""

import pytest

from pori.evaluation import Evaluator
from pori.memory import AgentMemory, ToolCallRecord


@pytest.fixture
def evaluator():
    return Evaluator(max_retries=3)


class TestEvaluateTaskCompletion:
    """Tests for evaluate_task_completion method."""

    def test_ignores_done_from_different_task(self, evaluator):
        """Evaluator should ignore done calls from a different task."""
        memory = AgentMemory()
        memory.current_task_id = "current_task"

        # Add a successful done from a different task
        old_done = ToolCallRecord(
            tool_name="done",
            parameters={"success": True, "message": "Old task done"},
            result={"message": "Old task done"},
            success=True,
            task_id="old_task",
        )
        memory.tool_call_history.append(old_done)

        is_complete, _ = evaluator.evaluate_task_completion("Test task", memory)

        # Should not be complete because the done call is from a different task
        assert is_complete is False

    def test_ignores_failed_answer_call(self, evaluator):
        """Evaluator should not count a failed answer call as task completion."""
        memory = AgentMemory()
        memory.current_task_id = "current_task"

        # Add a failed answer call
        failed_answer = ToolCallRecord(
            tool_name="answer",
            parameters={"final_answer": "Test", "reasoning": "Test"},
            result={"rejected": True, "message": "Rejected"},
            success=False,
            task_id="current_task",
        )
        memory.tool_call_history.append(failed_answer)

        is_complete, _ = evaluator.evaluate_task_completion("Test task", memory)

        # Should not be complete because the answer call failed
        assert is_complete is False

    def test_ignores_failed_done_call(self, evaluator):
        """Evaluator should not count a failed done call as task completion."""
        memory = AgentMemory()
        memory.current_task_id = "current_task"

        # Add a successful answer
        answer = ToolCallRecord(
            tool_name="answer",
            parameters={"final_answer": "42", "reasoning": "Because"},
            result={"final_answer": "42"},
            success=True,
            task_id="current_task",
        )
        memory.tool_call_history.append(answer)
        memory.update_state("final_answer", {"final_answer": "42"})

        # Add a failed done call
        failed_done = ToolCallRecord(
            tool_name="done",
            parameters={"success": True, "message": "Done"},
            result={"rejected": True, "message": "Rejected"},
            success=False,
            task_id="current_task",
        )
        memory.tool_call_history.append(failed_done)

        is_complete, message = evaluator.evaluate_task_completion("Test task", memory)

        # Should be complete (because of successful answer) but not due to done
        assert is_complete is True
        # The completion message should indicate answer, not "marked as complete"
        assert "Final answer provided" in message

    def test_counts_successful_answer_from_current_task(self, evaluator):
        """Evaluator should count a successful answer from the current task."""
        memory = AgentMemory()
        memory.current_task_id = "current_task"

        # Add a successful answer
        answer = ToolCallRecord(
            tool_name="answer",
            parameters={"final_answer": "42", "reasoning": "Because"},
            result={"final_answer": "42"},
            success=True,
            task_id="current_task",
        )
        memory.tool_call_history.append(answer)

        is_complete, _ = evaluator.evaluate_task_completion("Test task", memory)

        assert is_complete is True

    def test_counts_successful_done_from_current_task(self, evaluator):
        """Evaluator should count a successful done from the current task."""
        memory = AgentMemory()
        memory.current_task_id = "current_task"

        # Add a successful answer first
        answer = ToolCallRecord(
            tool_name="answer",
            parameters={"final_answer": "42", "reasoning": "Because"},
            result={"final_answer": "42"},
            success=True,
            task_id="current_task",
        )
        memory.tool_call_history.append(answer)
        memory.update_state("final_answer", {"final_answer": "42"})

        # Add a successful done
        done = ToolCallRecord(
            tool_name="done",
            parameters={"success": True, "message": "Task complete"},
            result={"message": "Task complete"},
            success=True,
            task_id="current_task",
        )
        memory.tool_call_history.append(done)

        is_complete, message = evaluator.evaluate_task_completion("Test task", memory)

        assert is_complete is True
        assert "marked as complete" in message.lower()

    def test_requires_answer_before_done_for_full_completion(self, evaluator):
        """A done without an answer should not result in full completion."""
        memory = AgentMemory()
        memory.current_task_id = "current_task"

        # Only add a successful done (no answer)
        done = ToolCallRecord(
            tool_name="done",
            parameters={"success": True, "message": "Task complete"},
            result={"message": "Task complete"},
            success=True,
            task_id="current_task",
        )
        memory.tool_call_history.append(done)

        is_complete, _ = evaluator.evaluate_task_completion("Test task", memory)

        # Should not be complete because no answer was provided
        assert is_complete is False

    def test_final_answer_state_triggers_completion(self, evaluator):
        """Setting final_answer state should trigger completion."""
        memory = AgentMemory()
        memory.current_task_id = "current_task"
        memory.update_state("final_answer", {"final_answer": "42"})

        is_complete, _ = evaluator.evaluate_task_completion("Test task", memory)

        assert is_complete is True
