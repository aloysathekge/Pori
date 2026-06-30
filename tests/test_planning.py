"""Tests for the model-driven planning tool and PlanStore."""

import asyncio

import pytest

from pori.agent import Agent, AgentOutput, AgentSettings
from pori.memory import AgentMemory
from pori.planning import PlanStore
from pori.tools.registry import ToolExecutor, ToolRegistry
from pori.tools.standard import STANDARD_KERNEL_TOOLS, register_all_tools


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def registry():
    r = ToolRegistry()
    register_all_tools(r)
    return r


# --- PlanStore --------------------------------------------------------------


def test_replace_dedupes_by_id_and_drops_empty():
    store = PlanStore()
    store.write(
        [
            {"id": "a", "content": "first"},
            {"content": ""},  # dropped
            {"id": "a", "content": "first revised"},  # same id, last wins
            {"content": "second"},
        ]
    )
    items = store.items()
    assert [i.content for i in items] == ["first revised", "second"]


def test_merge_updates_by_id_and_appends():
    store = PlanStore()
    store.write([{"id": "1", "content": "write", "status": "in_progress"}])
    store.write(
        [
            {"id": "1", "content": "write", "status": "completed"},
            {"id": "2", "content": "run"},
        ],
        merge=True,
    )
    assert {i.id: i.status for i in store.items()} == {"1": "completed", "2": "pending"}


def test_active_and_prompt_exclude_finished():
    store = PlanStore()
    store.write(
        [
            {"id": "1", "content": "done step", "status": "completed"},
            {"id": "2", "content": "current", "status": "in_progress"},
            {"id": "3", "content": "later", "status": "pending"},
            {"id": "4", "content": "dropped", "status": "cancelled"},
        ]
    )
    assert [i.content for i in store.active()] == ["current", "later"]
    rendered = store.format_for_prompt()
    assert "done step" not in rendered and "dropped" not in rendered
    assert "[>] current" in rendered and "[ ] later" in rendered


def test_invalid_status_defaults_to_pending():
    store = PlanStore()
    store.write([{"content": "x", "status": "bogus"}])
    assert store.items()[0].status == "pending"


# --- update_plan tool -------------------------------------------------------


def test_update_plan_tool_registered_in_kernel(registry):
    assert "update_plan" in STANDARD_KERNEL_TOOLS
    assert "update_plan" in registry.snapshot().tool_names


def test_system_prompt_nudges_update_plan():
    """The agent system prompt must steer the model to plan multi-step work."""
    from pori.utils.prompt_loader import load_prompt

    prompt = load_prompt("system/agent_core.md")
    assert "update_plan" in prompt


def test_update_plan_tool_writes_to_store(registry):
    store = PlanStore()
    result = ToolExecutor(registry).execute_tool(
        "update_plan",
        {"todos": [{"content": "step one", "status": "in_progress"}]},
        {"plan_store": store},
    )
    assert result["success"] is True
    assert result["result"]["summary"]["in_progress"] == 1
    assert store.items()[0].content == "step one"


def test_update_plan_tool_without_store_is_graceful(registry):
    result = ToolExecutor(registry).execute_tool(
        "update_plan", {"todos": [{"content": "x"}]}, {}
    )
    assert result["result"]["available"] is False


# --- agent integration ------------------------------------------------------


class _MockResp:
    def __init__(self, parsed):
        self.parsed = parsed

    def get(self, key, default=None):
        return self.parsed if key == "parsed" else default


class _MockLLM:
    def __init__(self, actions_per_call):
        self._actions = actions_per_call
        self.i = 0

    def with_structured_output(self, output_model, include_raw=True):
        return self

    async def ainvoke_tools(self, messages, tools):
        from tests._native_mock import tool_turn_from_response

        return tool_turn_from_response(await self.ainvoke(messages))

    async def ainvoke(self, messages):
        actions = self._actions[min(self.i, len(self._actions) - 1)]
        self.i += 1
        return _MockResp(
            AgentOutput(
                current_state={
                    "evaluation_previous_goal": "ok",
                    "memory": "none",
                    "next_goal": "Answering the user",
                },
                action=actions,
            )
        )


def test_agent_default_makes_no_side_planning_call(registry, event_loop):
    """With the default (model-driven) settings, the agent owns its plan store."""
    llm = _MockLLM(
        [
            [{"update_plan": {"todos": [{"content": "answer the user"}]}}],
            [{"answer": {"final_answer": "done", "reasoning": "r"}}],
        ]
    )
    agent = Agent(
        task="A multi step task",
        llm=llm,
        tools_registry=registry,
        settings=AgentSettings(max_steps=4),
        memory=AgentMemory(),
    )
    assert agent.settings.planning_mode == "never"
    result = event_loop.run_until_complete(agent.run())
    assert result["completed"] is True
    assert agent.plan_store.items()[0].content == "answer the user"
    # The plan is exposed in the run-result contract (for Cloud/UI consumers).
    assert result["plan"] == [
        {"id": "1", "content": "answer the user", "status": "pending"}
    ]
    assert agent.result_summary()["plan"] == result["plan"]
    # The model's next_goal is captured as the live activity line.
    assert agent.state.current_activity == "Answering the user"
