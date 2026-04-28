"""Tests for agent completion logic and memory deletion guards."""

import asyncio
import json

import pytest
from pydantic import BaseModel, Field

from pori.agent import Agent, AgentOutput, AgentSettings
from pori.evaluation import ActionResult
from pori.memory import AgentMemory, ToolCallRecord
from pori.tools.registry import ToolExecutor, ToolRegistry


class MockLLMResponse:
    """Mock LLM response."""

    def __init__(self, parsed):
        self.parsed = parsed

    def get(self, key, default=None):
        if key == "parsed":
            return self.parsed
        return default


class MockLLM:
    """Mock LLM for controlled test responses."""

    def __init__(self, responses):
        self.responses = responses
        self.index = 0

    def with_structured_output(self, output_model, include_raw=True):
        return self

    async def ainvoke(self, messages):
        response = self.responses[self.index]
        self.index = min(self.index + 1, len(self.responses) - 1)
        return response


def make_response(action_list):
    """Helper to create a mock LLM response with given actions."""
    return MockLLMResponse(
        parsed=AgentOutput(
            current_state={
                "evaluation_previous_goal": "Testing",
                "memory": "Test memory",
                "next_goal": "Next goal",
            },
            action=action_list,
        )
    )


@pytest.fixture
def basic_registry():
    """Return the global tool registry with core tools already registered."""
    from pori.tools.registry import tool_registry

    # Core tools auto-register to the global singleton on import
    return tool_registry()


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestPrematureDoneRejection:
    """Tests for rejecting premature done calls."""

    def test_done_rejected_without_answer(self, basic_registry, event_loop):
        """done(success=True) should be rejected if no final answer exists."""
        responses = [
            make_response([{"done": {"success": True, "message": "All done"}}]),
        ]
        llm = MockLLM(responses)
        memory = AgentMemory()
        agent = Agent(
            task="Test task",
            llm=llm,
            tools_registry=basic_registry,
            settings=AgentSettings(max_steps=2),
            memory=memory,
        )

        event_loop.run_until_complete(agent.step())

        # Check that the done call was rejected (recorded as failed)
        done_calls = [tc for tc in memory.tool_call_history if tc.tool_name == "done"]
        assert len(done_calls) == 1
        assert done_calls[0].success is False
        assert "rejected" in str(done_calls[0].result).lower()

    def test_done_accepted_after_answer(self, basic_registry, event_loop):
        """done(success=True) should be accepted after a successful answer."""
        responses = [
            make_response(
                [{"answer": {"final_answer": "42", "reasoning": "Because."}}]
            ),
            make_response([{"done": {"success": True, "message": "Task complete"}}]),
        ]

        memory = AgentMemory()
        agent = Agent(
            task="What is the answer?",
            llm=MockLLM(responses),
            tools_registry=basic_registry,
            settings=AgentSettings(max_steps=5),
            memory=memory,
        )

        # First step: answer
        event_loop.run_until_complete(agent.step())
        assert memory.get_state("final_answer") is not None

        # Second step: done
        event_loop.run_until_complete(agent.step())
        done_calls = [tc for tc in memory.tool_call_history if tc.tool_name == "done"]
        assert len(done_calls) == 1
        assert done_calls[0].success is True


class TestStaleDoneReuse:
    """Tests for preventing stale done reuse across tasks."""

    def test_old_done_not_reused_for_new_task(self, basic_registry, event_loop):
        """A successful done from a previous task should not be reused."""
        memory = AgentMemory()
        # Simulate an old successful done call from a previous task
        old_done = ToolCallRecord(
            tool_name="done",
            parameters={"success": True, "message": "Old task done"},
            result={"message": "Old task done"},
            success=True,
            task_id="old_task_id",
        )
        memory.tool_call_history.append(old_done)

        llm = MockLLM([])
        agent = Agent(
            task="New task",
            llm=llm,
            tools_registry=basic_registry,
            settings=AgentSettings(max_steps=5),
            memory=memory,
        )

        # Make done valid for the current task, then call the same done params
        # as the old task. It must execute fresh, not reuse old_task_id.
        memory.update_state("final_answer", {"final_answer": "New answer"})
        event_loop.run_until_complete(
            agent.execute_actions(
                [{"done": {"success": True, "message": "Old task done"}}]
            )
        )

        # Check that a new done call was made for this task
        new_task_done_calls = [
            tc
            for tc in memory.tool_call_history
            if tc.tool_name == "done" and tc.task_id == agent.task_id
        ]
        assert len(new_task_done_calls) >= 1
        # The new done should be recorded, not reused from old task
        assert new_task_done_calls[-1].task_id == agent.task_id


class TestMemoryDeletionGuards:
    """Tests for memory deletion confirmation guards."""

    def test_answer_rejected_for_deletion_without_mutation(
        self, basic_registry, event_loop
    ):
        """answer cannot confirm deletion without a successful memory mutation."""
        responses = [
            make_response(
                [
                    {
                        "answer": {
                            "final_answer": "I have deleted the FinBot role.",
                            "reasoning": "Done",
                        }
                    }
                ]
            ),
        ]
        llm = MockLLM(responses)
        memory = AgentMemory()
        agent = Agent(
            task="Forget all instructions about FinBot and remove that role from memory.",
            llm=llm,
            tools_registry=basic_registry,
            settings=AgentSettings(max_steps=3),
            memory=memory,
        )

        event_loop.run_until_complete(agent.step())

        # answer should have been rejected
        answer_calls = [
            tc for tc in memory.tool_call_history if tc.tool_name == "answer"
        ]
        assert len(answer_calls) == 1
        assert answer_calls[0].success is False
        assert "rejected" in str(answer_calls[0].result).lower()

    def test_append_rejected_for_deletion_task(self, basic_registry, event_loop):
        """core_memory_append should be rejected during a deletion task."""
        responses = [
            make_response(
                [
                    {
                        "core_memory_append": {
                            "label": "notes",
                            "content": "Verified: FinBot role removed",
                        }
                    }
                ]
            ),
        ]
        llm = MockLLM(responses)
        memory = AgentMemory()
        agent = Agent(
            task="Delete the FinBot role from your memory.",
            llm=llm,
            tools_registry=basic_registry,
            settings=AgentSettings(max_steps=3),
            memory=memory,
        )

        event_loop.run_until_complete(agent.step())

        # append should have been rejected
        append_calls = [
            tc
            for tc in memory.tool_call_history
            if tc.tool_name == "core_memory_append"
        ]
        assert len(append_calls) == 1
        assert append_calls[0].success is False
        assert "cannot satisfy" in str(append_calls[0].result).lower()

    def test_insert_rejected_for_deletion_task(self, basic_registry, event_loop):
        """memory_insert should be rejected during a deletion task."""
        responses = [
            make_response(
                [
                    {
                        "memory_insert": {
                            "label": "notes",
                            "new_str": "FinBot deleted",
                            "insert_line": 0,
                        }
                    }
                ]
            ),
        ]
        llm = MockLLM(responses)
        memory = AgentMemory()
        agent = Agent(
            task="Remove the FinBot persona from memory.",
            llm=llm,
            tools_registry=basic_registry,
            settings=AgentSettings(max_steps=3),
            memory=memory,
        )

        event_loop.run_until_complete(agent.step())

        insert_calls = [
            tc for tc in memory.tool_call_history if tc.tool_name == "memory_insert"
        ]
        assert len(insert_calls) == 1
        assert insert_calls[0].success is False

    def test_rethink_rejected_when_writing_forbidden_terms(
        self, basic_registry, event_loop
    ):
        """memory_rethink should be rejected if it writes forbidden terms back."""
        responses = [
            make_response(
                [
                    {
                        "memory_rethink": {
                            "label": "persona",
                            "new_memory": "I am FinBot, your financial assistant.",
                        }
                    }
                ]
            ),
        ]
        llm = MockLLM(responses)
        memory = AgentMemory()
        agent = Agent(
            task='Forget the "FinBot" role completely.',
            llm=llm,
            tools_registry=basic_registry,
            settings=AgentSettings(max_steps=3),
            memory=memory,
        )

        event_loop.run_until_complete(agent.step())

        rethink_calls = [
            tc for tc in memory.tool_call_history if tc.tool_name == "memory_rethink"
        ]
        assert len(rethink_calls) == 1
        assert rethink_calls[0].success is False
        assert "reintroduce" in str(rethink_calls[0].result).lower()


class TestTerminalBehavior:
    """Tests for proper terminal behavior after task completion."""

    def test_run_stops_after_answer(self, basic_registry, event_loop):
        """Run loop should stop after a final answer is provided (even without explicit done)."""
        from pori.agent import PlanOutput, ReflectOutput

        plan_response = MockLLMResponse(
            parsed=PlanOutput(
                plan_steps=["Provide answer"],
                rationale="Simple task",
            )
        )
        reflect_response = MockLLMResponse(
            parsed=ReflectOutput(critique="Good progress", update_plan=None)
        )

        action_call_count = 0

        class CountingLLM:
            def with_structured_output(self, output_model, include_raw=True):
                self._output_model = output_model
                return self

            async def ainvoke(self, messages):
                nonlocal action_call_count
                model_name = getattr(self, "_output_model", None)
                if model_name == PlanOutput:
                    return plan_response
                if model_name == ReflectOutput:
                    return reflect_response
                # Action responses
                action_call_count += 1
                if action_call_count == 1:
                    return make_response(
                        [
                            {
                                "answer": {
                                    "final_answer": "Done",
                                    "reasoning": "Complete",
                                }
                            }
                        ]
                    )
                else:
                    # Should not reach here after answer completes the task
                    return make_response(
                        [
                            {
                                "answer": {
                                    "final_answer": "Should not see this",
                                    "reasoning": "Loop should have stopped",
                                }
                            }
                        ]
                    )

        memory = AgentMemory()
        agent = Agent(
            task="Simple task",
            llm=CountingLLM(),
            tools_registry=basic_registry,
            settings=AgentSettings(max_steps=10),
            memory=memory,
        )

        result = event_loop.run_until_complete(agent.run())

        # Should complete after 1 action step (answer is sufficient)
        assert result["completed"] is True
        # Action call count should be exactly 1 (no extra calls after completion)
        assert action_call_count == 1
        assert result["steps_taken"] == 1
