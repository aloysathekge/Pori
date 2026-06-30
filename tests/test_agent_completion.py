"""Tests for agent completion logic and memory deletion guards."""

import asyncio
import json

import pytest
from pydantic import BaseModel, Field

from pori.agent import Agent, AgentOutput, AgentSettings
from pori.evaluation import ActionResult
from pori.memory import AgentMemory, ToolCallRecord
from pori.runtime import ReceiptStatus
from pori.tools.policy import ToolAuthorizationPolicy
from pori.tools.registry import ToolExecutor, ToolRegistry
from pori.tools.standard.filesystem_tools import fs_config


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

    def test_result_summary_reports_written_artifacts(
        self, basic_registry, event_loop, tmp_path, monkeypatch
    ):
        # Allow writes under tmp_path on every platform (CI tmp dirs live outside
        # home/cwd, which the filesystem-safety check would otherwise reject).
        monkeypatch.setattr(fs_config, "current_dir", tmp_path.resolve())
        memory = AgentMemory()
        agent = Agent(
            task="Create a lesson file",
            llm=MockLLM([]),
            tools_registry=basic_registry,
            settings=AgentSettings(max_steps=2),
            memory=memory,
        )
        path = tmp_path / "lesson.html"

        event_loop.run_until_complete(
            agent.execute_actions(
                [
                    {
                        "write_file": {
                            "file_path": str(path),
                            "content": "<html>lesson</html>",
                        }
                    }
                ]
            )
        )

        artifacts = agent.result_summary()["artifacts"]
        assert len(artifacts) == 1
        assert artifacts[0] == {
            "kind": "file",
            "tool_name": "write_file",
            "path": str(path),
            "operation": "write",
            "bytes_written": 19,
            "receipt_id": artifacts[0]["receipt_id"],
        }
        assert artifacts[0]["receipt_id"].startswith("rcpt_")

    def test_file_write_allowed_by_default(
        self, basic_registry, event_loop, tmp_path, monkeypatch
    ):
        """By default the model is trusted to write files; receipts keep it honest."""
        monkeypatch.setattr(fs_config, "current_dir", tmp_path.resolve())
        memory = AgentMemory()
        agent = Agent(
            task="Teach me division step by step simply",
            llm=MockLLM([]),
            tools_registry=basic_registry,
            settings=AgentSettings(max_steps=2),
            memory=memory,
        )
        path = tmp_path / "styles.css"

        event_loop.run_until_complete(
            agent.execute_actions(
                [
                    {
                        "write_file": {
                            "file_path": str(path),
                            "content": "body { color: red; }",
                        }
                    }
                ]
            )
        )

        write_calls = [
            tc for tc in memory.tool_call_history if tc.tool_name == "write_file"
        ]
        assert len(write_calls) == 1
        assert write_calls[0].success is True
        assert path.exists()

    def test_strict_policy_blocks_file_write_without_request(
        self, basic_registry, event_loop, tmp_path
    ):
        """Opt-in strict mode still blocks writes the task did not ask for."""
        memory = AgentMemory()
        agent = Agent(
            task="Teach me division step by step simply",
            llm=MockLLM([]),
            tools_registry=basic_registry,
            settings=AgentSettings(max_steps=2),
            memory=memory,
            tool_authorization_policy=ToolAuthorizationPolicy(
                require_artifact_intent=True
            ),
        )
        path = tmp_path / "lessons"

        event_loop.run_until_complete(
            agent.execute_actions(
                [
                    {
                        "create_directory": {
                            "directory_path": str(path),
                            "parents": True,
                        }
                    }
                ]
            )
        )

        directory_calls = [
            tc for tc in memory.tool_call_history if tc.tool_name == "create_directory"
        ]
        assert len(directory_calls) == 1
        assert directory_calls[0].success is False
        assert "did not explicitly ask for a file artifact" in str(
            directory_calls[0].result
        )
        assert not path.exists()

    def test_answer_prose_file_claim_is_not_a_runtime_artifact(
        self, basic_registry, event_loop
    ):
        """Prose alone is not the source of truth for UI-visible artifacts."""
        memory = AgentMemory()
        agent = Agent(
            task="teach me division and create an HTML lesson file",
            llm=MockLLM([]),
            tools_registry=basic_registry,
            settings=AgentSettings(max_steps=2),
            memory=memory,
        )

        event_loop.run_until_complete(
            agent.execute_actions(
                [
                    {
                        "answer": {
                            "final_answer": (
                                "Your division lesson is ready! I created an "
                                "interactive HTML file at lessons/division.html."
                            ),
                            "reasoning": "The requested file has been created.",
                        }
                    }
                ]
            )
        )

        answer_calls = [
            tc for tc in memory.tool_call_history if tc.tool_name == "answer"
        ]
        assert len(answer_calls) == 1
        assert answer_calls[0].success is True
        assert memory.get_state("final_answer") is not None
        assert agent.result_summary()["artifacts"] == []

    def test_answer_rejects_artifact_reference_without_write_receipt(
        self, basic_registry, event_loop
    ):
        """UI-visible artifact references must match runtime receipts."""
        memory = AgentMemory()
        agent = Agent(
            task="teach me division and create an HTML lesson file",
            llm=MockLLM([]),
            tools_registry=basic_registry,
            settings=AgentSettings(max_steps=2),
            memory=memory,
        )

        event_loop.run_until_complete(
            agent.execute_actions(
                [
                    {
                        "answer": {
                            "final_answer": (
                                "Your division lesson is ready! I created an "
                                "interactive HTML file at lessons/division.html."
                            ),
                            "reasoning": "The requested file has been created.",
                            "artifact_references": [{"path": "lessons/division.html"}],
                        }
                    }
                ]
            )
        )

        answer_calls = [
            tc for tc in memory.tool_call_history if tc.tool_name == "answer"
        ]
        assert len(answer_calls) == 1
        assert answer_calls[0].success is False
        assert "Invalid artifact_references" in str(answer_calls[0].result)
        assert memory.get_state("final_answer") is None

    def test_answer_accepts_artifact_reference_after_write_receipt(
        self, basic_registry, event_loop
    ):
        """A successful current-task write_file receipt satisfies artifact refs."""
        memory = AgentMemory()
        agent = Agent(
            task="teach me division and create an HTML lesson file",
            llm=MockLLM([]),
            tools_registry=basic_registry,
            settings=AgentSettings(max_steps=2),
            memory=memory,
        )
        receipt = agent._record_tool_receipt(
            "write_file",
            {"file_path": "lessons/division.html", "content": "<html></html>"},
            ReceiptStatus.SUCCEEDED,
            artifacts=[
                {
                    "kind": "file",
                    "tool_name": "write_file",
                    "path": "lessons/division.html",
                    "operation": "write",
                }
            ],
        )

        event_loop.run_until_complete(
            agent.execute_actions(
                [
                    {
                        "answer": {
                            "final_answer": (
                                "Your division lesson is ready. See the attached "
                                "artifact."
                            ),
                            "reasoning": "The runtime receipt proves the artifact.",
                            "artifact_references": [
                                {
                                    "path": "lessons/division.html",
                                    "receipt_id": receipt.receipt_id,
                                }
                            ],
                        }
                    }
                ]
            )
        )

        answer_calls = [
            tc for tc in memory.tool_call_history if tc.tool_name == "answer"
        ]
        assert answer_calls[-1].success is True
        final = memory.get_state("final_answer")
        assert final is not None
        assert final["artifact_references"] == [
            {
                "path": "lessons/division.html",
                "receipt_id": receipt.receipt_id,
            }
        ]

    def test_answer_accepts_receipt_id_only_reference(self, basic_registry, event_loop):
        """Referencing an artifact by receipt_id alone (no path) is valid."""
        memory = AgentMemory()
        agent = Agent(
            task="create an HTML lesson file",
            llm=MockLLM([]),
            tools_registry=basic_registry,
            settings=AgentSettings(max_steps=2),
            memory=memory,
        )
        receipt = agent._record_tool_receipt(
            "write_file",
            {"file_path": "lessons/division.html", "content": "<html></html>"},
            ReceiptStatus.SUCCEEDED,
            artifacts=[{"kind": "file", "path": "lessons/division.html"}],
        )

        event_loop.run_until_complete(
            agent.execute_actions(
                [
                    {
                        "answer": {
                            "final_answer": "Lesson ready.",
                            "reasoning": "Receipt proves it.",
                            "artifact_references": [
                                {
                                    "receipt_id": receipt.receipt_id,
                                    "description": "vowels script",
                                }
                            ],
                        }
                    }
                ]
            )
        )

        answer_calls = [
            tc for tc in memory.tool_call_history if tc.tool_name == "answer"
        ]
        assert answer_calls[-1].success is True
        assert memory.get_state("final_answer") is not None

    def test_answer_rejects_reference_with_neither_path_nor_receipt(
        self, basic_registry, event_loop
    ):
        """An empty reference is rejected gracefully, not via a hard ValidationError."""
        memory = AgentMemory()
        agent = Agent(
            task="create an HTML lesson file",
            llm=MockLLM([]),
            tools_registry=basic_registry,
            settings=AgentSettings(max_steps=2),
            memory=memory,
        )

        event_loop.run_until_complete(
            agent.execute_actions(
                [
                    {
                        "answer": {
                            "final_answer": "Done.",
                            "reasoning": "No real artifact.",
                            "artifact_references": [{"description": "a file"}],
                        }
                    }
                ]
            )
        )

        answer_calls = [
            tc for tc in memory.tool_call_history if tc.tool_name == "answer"
        ]
        assert answer_calls[-1].success is False
        assert "Invalid artifact_references" in str(answer_calls[-1].result)
        assert memory.get_state("final_answer") is None

    def test_answer_accepts_reference_to_one_of_many_receipt_artifacts(
        self, basic_registry, event_loop
    ):
        """A receipt with multiple artifacts validates any of its paths."""
        memory = AgentMemory()
        agent = Agent(
            task="create the lesson files",
            llm=MockLLM([]),
            tools_registry=basic_registry,
            settings=AgentSettings(max_steps=2),
            memory=memory,
        )
        receipt = agent._record_tool_receipt(
            "write_file",
            {"file_path": "lessons/", "content": "..."},
            ReceiptStatus.SUCCEEDED,
            artifacts=[
                {"kind": "file", "path": "lessons/a.html", "operation": "write"},
                {"kind": "file", "path": "lessons/b.html", "operation": "write"},
            ],
        )

        # Reference the FIRST artifact by path + receipt_id (the old code matched
        # only the last artifact per receipt and would falsely reject this).
        event_loop.run_until_complete(
            agent.execute_actions(
                [
                    {
                        "answer": {
                            "final_answer": "Lessons ready.",
                            "reasoning": "Both files written under one receipt.",
                            "artifact_references": [
                                {
                                    "path": "lessons/a.html",
                                    "receipt_id": receipt.receipt_id,
                                }
                            ],
                        }
                    }
                ]
            )
        )

        answer_calls = [
            tc for tc in memory.tool_call_history if tc.tool_name == "answer"
        ]
        assert answer_calls[-1].success is True
        assert memory.get_state("final_answer") is not None

    def test_get_next_action_retries_invalid_structured_json(
        self, basic_registry, event_loop
    ):
        """A truncated structured response should get one corrective retry."""

        class RetryLLM:
            def __init__(self):
                self.calls = 0
                self.messages = []
                self._output_model = None

            def with_structured_output(self, output_model, include_raw=True):
                self._output_model = output_model
                return self

            async def ainvoke(self, messages):
                self.calls += 1
                self.messages.append(messages)
                if self.calls == 1:
                    return {
                        "parsed": None,
                        "raw": '{"current_state": {"memory": "cut"',
                    }
                return {
                    "parsed": self._output_model(
                        current_state={
                            "evaluation_previous_goal": "Recovered",
                            "memory": "Retried after invalid JSON",
                            "next_goal": "Answer",
                        },
                        action=[
                            {
                                "answer": {
                                    "final_answer": "1 + 1 = 2",
                                    "reasoning": "Recovered structured output",
                                }
                            }
                        ],
                    ),
                    "raw": None,
                }

        llm = RetryLLM()
        memory = AgentMemory()
        agent = Agent(
            task="Teach basic 1+1",
            llm=llm,
            tools_registry=basic_registry,
            settings=AgentSettings(max_steps=2),
            memory=memory,
        )

        output = event_loop.run_until_complete(agent.get_next_action())

        assert llm.calls == 2
        assert "invalid or incomplete JSON" in llm.messages[-1][-1].content
        assert output.action[0]["answer"]["final_answer"] == "1 + 1 = 2"

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


def test_sandbox_thread_keyed_to_session(basic_registry):
    """Two tasks in the same session share one sandbox workspace."""
    memory = AgentMemory()
    a1 = Agent(
        task="first task",
        llm=MockLLM([]),
        tools_registry=basic_registry,
        settings=AgentSettings(max_steps=2),
        memory=memory,
    )
    a2 = Agent(
        task="second task",
        llm=MockLLM([]),
        tools_registry=basic_registry,
        settings=AgentSettings(max_steps=2),
        memory=memory,
    )

    # Distinct tasks, but the sandbox is keyed to the shared session.
    assert a1.task_id != a2.task_id
    assert a1._sandbox_thread_id() == a2._sandbox_thread_id()
    assert a1._sandbox_thread_id() == a1.run_context.session_id
