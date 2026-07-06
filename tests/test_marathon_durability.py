"""Phase 1 marathon-durability tests: write-ahead tool journal, per-step
loop checkpointing, resume-from-checkpoint, and the salvage summary.

See docs/long-running.md ("never lose work").
"""

import asyncio

import pytest

from pori.agent import Agent, AgentSettings
from pori.agent.schemas import AgentOutput
from pori.memory import AgentMemory, InMemoryMemoryStore


class MockLLMResponse:
    def __init__(self, parsed):
        self.parsed = parsed

    def get(self, key, default=None):
        if key == "parsed":
            return self.parsed
        return default


class MockLLM:
    """Serves scripted tool-calling turns; plain ainvoke returns a string
    (exercises the salvage-summary path)."""

    def __init__(self, responses, plain_response="Salvaged summary of progress."):
        self.responses = responses
        self.index = 0
        self.plain_response = plain_response
        self.plain_calls = 0

    def with_structured_output(self, output_model, include_raw=True):
        return self

    async def ainvoke(self, messages):
        self.plain_calls += 1
        return self.plain_response

    async def ainvoke_tools(self, messages, tools):
        from pori.llm import ToolTurn
        from tests._native_mock import tool_turn_from_response

        if not self.responses:
            return ToolTurn()
        response = self.responses[self.index]
        self.index = min(self.index + 1, len(self.responses) - 1)
        return tool_turn_from_response(response)


def make_response(action_list, next_goal="working"):
    return MockLLMResponse(
        parsed=AgentOutput(
            current_state={
                "evaluation_previous_goal": "ok",
                "memory": "none",
                "next_goal": next_goal,
            },
            action=action_list,
        )
    )


@pytest.fixture
def registry():
    from pori.tools.registry import tool_registry
    from pori.tools.standard.planning_tools import register_planning_tools

    reg = tool_registry()
    try:
        reg.get_tool("update_plan")
    except ValueError:
        register_planning_tools(reg)
    return reg


ANSWER = make_response([{"answer": {"final_answer": "42", "reasoning": "done"}}])
DONE = make_response([{"done": {"success": True, "message": "finished"}}])
PLAN = make_response(
    [
        {
            "update_plan": {
                "todos": [
                    {"id": "1", "content": "first thing", "status": "in_progress"},
                    {"id": "2", "content": "second thing", "status": "pending"},
                ]
            }
        }
    ]
)


class TestWriteAheadJournal:
    async def test_tool_call_journaled_before_and_completed_after(self, registry):
        memory = AgentMemory()
        agent = Agent(
            task="answer the question",
            llm=MockLLM([ANSWER]),
            tools_registry=registry,
            settings=AgentSettings(max_steps=2),
            memory=memory,
        )
        await agent.step()

        answer_calls = [
            tc for tc in memory.tool_call_history if tc.tool_name == "answer"
        ]
        assert len(answer_calls) == 1
        # The record went through dispatched -> completed and carries the result
        assert answer_calls[0].status == "completed"
        assert answer_calls[0].success is True
        assert memory.pending_dispatches(agent.task_id) == []

    async def test_crash_between_dispatch_and_completion_is_visible(self, registry):
        store = InMemoryMemoryStore()
        memory = AgentMemory(session_id="crashy", store=store)
        memory.create_task("task-1", "do something")
        # Simulate the crash window: dispatch persisted, process dies before
        # completion. A reload from the same store must surface it.
        memory.record_tool_dispatch("write_file", {"path": "out.txt"})

        reloaded = AgentMemory(session_id="crashy", store=store)
        pending = reloaded.pending_dispatches("task-1")
        assert len(pending) == 1
        assert pending[0].tool_name == "write_file"
        assert pending[0].status == "dispatched"

    async def test_tool_exception_closes_the_journal(self, registry, monkeypatch):
        memory = AgentMemory()
        agent = Agent(
            task="answer",
            llm=MockLLM([ANSWER]),
            tools_registry=registry,
            settings=AgentSettings(max_steps=2),
            memory=memory,
        )

        def boom(**kwargs):
            raise RuntimeError("executor exploded")

        monkeypatch.setattr(agent.tool_executor, "execute_tool", boom)
        await agent.step()
        # The process survived the failure, so nothing may linger as dispatched
        assert memory.pending_dispatches(agent.task_id) == []


class TestLoopCheckpoint:
    async def test_step_checkpoints_progress_and_plan(self, registry):
        memory = AgentMemory()
        agent = Agent(
            task="plan then answer",
            llm=MockLLM([PLAN, ANSWER]),
            tools_registry=registry,
            settings=AgentSettings(max_steps=5),
            memory=memory,
        )
        await agent.step()

        record = memory.tasks[agent.task_id]
        assert record.n_steps == 1
        assert record.progress_updated_at is not None
        assert [item["content"] for item in record.plan] == [
            "first thing",
            "second thing",
        ]

    async def test_checkpoint_survives_reload(self, registry):
        store = InMemoryMemoryStore()
        memory = AgentMemory(session_id="persist", store=store)
        agent = Agent(
            task="plan things",
            llm=MockLLM([PLAN]),
            tools_registry=registry,
            settings=AgentSettings(max_steps=5),
            memory=memory,
        )
        await agent.step()

        reloaded = AgentMemory(session_id="persist", store=store)
        record = reloaded.tasks[agent.task_id]
        assert record.n_steps == 1
        assert len(record.plan) == 2


class TestResume:
    async def test_resume_restores_position_and_plan(self, registry):
        store = InMemoryMemoryStore()
        memory = AgentMemory(session_id="resume", store=store)
        first = Agent(
            task="long task",
            llm=MockLLM([PLAN]),
            tools_registry=registry,
            settings=AgentSettings(max_steps=10),
            memory=memory,
        )
        await first.step()
        assert first.state.n_steps == 1

        # "Restart": fresh memory from the same store, resume the same task
        reloaded = AgentMemory(session_id="resume", store=store)
        second = Agent(
            task="long task",
            llm=MockLLM([ANSWER, DONE]),
            tools_registry=registry,
            settings=AgentSettings(max_steps=10),
            memory=reloaded,
            resume_task_id=first.task_id,
        )
        assert second.task_id == first.task_id
        assert second.state.n_steps == 1
        assert [item.content for item in second.plan_store.items()] == [
            "first thing",
            "second thing",
        ]
        # No duplicate task record was created
        assert list(reloaded.tasks.keys()).count(first.task_id) == 1

        result = await second.run()
        assert result["completed"] is True
        # Total steps span both processes
        assert result["steps_taken"] >= 2

    async def test_resume_surfaces_interrupted_dispatches(self, registry):
        store = InMemoryMemoryStore()
        memory = AgentMemory(session_id="interrupted", store=store)
        first = Agent(
            task="risky task",
            llm=MockLLM([PLAN]),
            tools_registry=registry,
            settings=AgentSettings(max_steps=10),
            memory=memory,
        )
        await first.step()
        # Crash mid-tool: a dispatch that never completes
        memory.record_tool_dispatch("write_file", {"path": "half-written.txt"})

        reloaded = AgentMemory(session_id="interrupted", store=store)
        second = Agent(
            task="risky task",
            llm=MockLLM([ANSWER]),
            tools_registry=registry,
            settings=AgentSettings(max_steps=10),
            memory=reloaded,
            resume_task_id=first.task_id,
        )
        resume_notes = [
            m
            for m in reloaded.messages
            if m.role == "system" and "[resume]" in str(m.content)
        ]
        assert len(resume_notes) == 1
        assert "write_file" in str(resume_notes[0].content)

    async def test_resume_of_terminal_task_raises(self, registry):
        store = InMemoryMemoryStore()
        memory = AgentMemory(session_id="terminal", store=store)
        memory.create_task("done-task", "finished work")
        memory.tasks["done-task"].complete(success=True)
        memory.persist()

        with pytest.raises(ValueError, match="Cannot resume"):
            Agent(
                task="finished work",
                llm=MockLLM([ANSWER]),
                tools_registry=registry,
                settings=AgentSettings(max_steps=5),
                memory=AgentMemory(session_id="terminal", store=store),
                resume_task_id="done-task",
            )

    async def test_resume_with_unknown_task_id_starts_fresh_with_that_id(
        self, registry
    ):
        agent = Agent(
            task="new task",
            llm=MockLLM([ANSWER]),
            tools_registry=registry,
            settings=AgentSettings(max_steps=5),
            memory=AgentMemory(),
            resume_task_id="stable-external-id",
        )
        assert agent.task_id == "stable-external-id"
        assert agent.state.n_steps == 0
        assert "stable-external-id" in agent.memory.tasks


class TestSalvageSummary:
    async def test_exhausted_run_delivers_partial_result(self, registry):
        llm = MockLLM([PLAN], plain_response="Got halfway: made the plan.")
        agent = Agent(
            task="impossible task",
            llm=llm,
            tools_registry=registry,
            settings=AgentSettings(max_steps=1, salvage_summary=True),
            memory=AgentMemory(),
        )
        result = await agent.run()

        assert result["completed"] is False
        assert result["partial_result"] is not None
        assert result["partial_result"]["reason"] == "step_limit"
        assert "halfway" in result["partial_result"]["summary"]
        assert llm.plain_calls == 1
        # Persisted for later consumers too
        assert agent.memory.get_state("partial_result") is not None

    async def test_salvage_skipped_when_completed(self, registry):
        llm = MockLLM([ANSWER, DONE])
        agent = Agent(
            task="easy task",
            llm=llm,
            tools_registry=registry,
            settings=AgentSettings(max_steps=5),
            memory=AgentMemory(),
        )
        result = await agent.run()
        assert result["completed"] is True
        assert result["partial_result"] is None
        assert llm.plain_calls == 0

    async def test_salvage_disabled_by_setting(self, registry):
        llm = MockLLM([PLAN])
        agent = Agent(
            task="impossible task",
            llm=llm,
            tools_registry=registry,
            settings=AgentSettings(max_steps=1, salvage_summary=False),
            memory=AgentMemory(),
        )
        result = await agent.run()
        assert result["partial_result"] is None
        assert llm.plain_calls == 0

    async def test_duration_budget_enforced(self):
        import time as _time

        from pori.runtime import BudgetExceeded, BudgetLedger, ExecutionBudget

        ledger = BudgetLedger(ExecutionBudget(max_duration_seconds=0.01))
        ledger.start_clock()
        _time.sleep(0.05)
        with pytest.raises(BudgetExceeded, match="Duration"):
            ledger.consume_step()
        snapshot = ledger.snapshot()
        assert snapshot["max_duration_seconds"] == 0.01
        assert snapshot["duration_seconds_used"] > 0

    async def test_no_duration_budget_never_expires(self):
        from pori.runtime import BudgetLedger, ExecutionBudget

        ledger = BudgetLedger(ExecutionBudget())
        ledger.start_clock()
        ledger.consume_step()  # must not raise

    async def test_orchestrator_forwards_resume_task_id(self, registry):
        from pori import Orchestrator

        store = InMemoryMemoryStore()
        memory = AgentMemory(session_id="orch-resume", store=store)
        orchestrator = Orchestrator(
            llm=MockLLM([ANSWER, DONE]), tools_registry=registry
        )
        result = await orchestrator.execute_task(
            task="stable task",
            agent_settings=AgentSettings(max_steps=5),
            memory=memory,
            resume_task_id="stable-orch-id",
        )
        assert result["success"] is True
        # The kernel task ran under the caller-supplied stable id
        assert "stable-orch-id" in memory.tasks
        assert memory.tasks["stable-orch-id"].status == "completed"

    async def test_salvage_fails_open(self, registry):
        class ExplodingLLM(MockLLM):
            async def ainvoke(self, messages):
                raise RuntimeError("provider down")

        agent = Agent(
            task="impossible task",
            llm=ExplodingLLM([PLAN]),
            tools_registry=registry,
            settings=AgentSettings(max_steps=1, salvage_summary=True),
            memory=AgentMemory(),
        )
        result = await agent.run()
        assert result["completed"] is False
        assert result["partial_result"] is None
