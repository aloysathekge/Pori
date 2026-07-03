"""Sub-agent delegation, Hermes-style: prompt, runner (single/batch/depth), tool."""

import asyncio

import pytest

from pori.orchestrator.core import Orchestrator
from pori.subagents import (
    MAX_CONCURRENT_CHILDREN,
    build_child_system_prompt,
    make_delegate_runner,
)
from pori.tools.standard.core_tools import (
    DelegateTaskItem,
    DelegateTaskParams,
    delegate_task_tool,
)

pytestmark = [pytest.mark.unit]


# --- child prompt (goal + context + role) -----------------------------------
def test_child_prompt_includes_goal_and_context_no_delegation_for_leaf():
    prompt = build_child_system_prompt("do X", "some ctx", role="leaf")
    assert "YOUR TASK:\ndo X" in prompt
    assert "CONTEXT:\nsome ctx" in prompt
    assert "Delegation" not in prompt  # a leaf cannot delegate


def test_child_prompt_orchestrator_gets_delegation_block_within_depth():
    prompt = build_child_system_prompt(
        "do X", role="orchestrator", child_depth=1, max_depth=2
    )
    assert "Delegation" in prompt and "delegate_task" in prompt


def test_child_prompt_orchestrator_at_cap_has_no_delegation_block():
    prompt = build_child_system_prompt(
        "do X", role="orchestrator", child_depth=2, max_depth=2
    )
    assert "Delegation" not in prompt


# --- runner: single, batch, depth -------------------------------------------
def test_runner_single_task():
    class _Stub:
        async def run_subagent(self, goal, *, allow_delegation=False, **kw):
            return f"done:{goal}|deleg={allow_delegation}"

    result = make_delegate_runner(_Stub())([{"goal": "task A", "role": "leaf"}])
    assert result == [
        {"success": True, "role": "leaf", "result": "done:task A|deleg=False"}
    ]


def test_runner_batch_runs_concurrently():
    state = {"active": 0, "max": 0}

    class _Slow:
        async def run_subagent(self, goal, **kw):
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
            await asyncio.sleep(0.05)
            state["active"] -= 1
            return goal

    make_delegate_runner(_Slow())([{"goal": "a"}, {"goal": "b"}, {"goal": "c"}])
    assert state["max"] >= 2  # they overlapped -> genuinely concurrent


def test_runner_orchestrator_within_depth_gets_a_child_runner():
    seen = {}

    class _Stub:
        async def run_subagent(
            self, goal, *, allow_delegation=False, child_tool_context=None, **kw
        ):
            seen["allow"] = allow_delegation
            seen["has_child_runner"] = bool(
                child_tool_context
                and callable(child_tool_context.get("delegate_runner"))
            )
            return "ok"

    make_delegate_runner(_Stub(), child_depth=1, max_depth=2)(
        [{"goal": "x", "role": "orchestrator"}]
    )
    assert seen["allow"] is True and seen["has_child_runner"] is True


def test_runner_leaf_and_depth_cap_cannot_delegate():
    seen = {}

    class _Stub:
        async def run_subagent(
            self, goal, *, allow_delegation=False, child_tool_context=None, **kw
        ):
            seen[goal] = (allow_delegation, child_tool_context)
            return "ok"

    runner_leaf = make_delegate_runner(_Stub())
    runner_leaf([{"goal": "leaf-task", "role": "leaf"}])
    assert seen["leaf-task"] == (False, None)

    runner_capped = make_delegate_runner(_Stub(), child_depth=2, max_depth=2)
    runner_capped([{"goal": "capped", "role": "orchestrator"}])
    assert seen["capped"][0] is False  # at the depth cap -> forced leaf


def test_runner_empty_goal_is_a_per_task_error():
    class _Stub:
        async def run_subagent(self, goal, **kw):
            return "x"

    result = make_delegate_runner(_Stub())([{"goal": "   ", "role": "leaf"}])
    assert result[0]["success"] is False


# --- the delegate_task tool -------------------------------------------------
def test_delegate_task_tool_delegates_batch():
    def runner(items):
        return [
            {"success": True, "role": it["role"], "result": it["goal"]} for it in items
        ]

    res = delegate_task_tool(
        DelegateTaskParams(
            tasks=[DelegateTaskItem(goal="a"), DelegateTaskItem(goal="b")]
        ),
        {"delegate_runner": runner},
    )
    assert res["success"] is True and res["count"] == 2
    assert res["results"][0]["result"] == "a"


def test_delegate_task_tool_refuses_when_no_runner():
    res = delegate_task_tool(DelegateTaskParams(tasks=[DelegateTaskItem(goal="a")]), {})
    assert res["success"] is False and "not available" in res["error"].lower()


def test_delegate_task_tool_rejects_too_many():
    items = [DelegateTaskItem(goal=str(i)) for i in range(MAX_CONCURRENT_CHILDREN + 1)]
    res = delegate_task_tool(
        DelegateTaskParams(tasks=items), {"delegate_runner": lambda x: []}
    )
    assert res["success"] is False and "max" in res["error"].lower()


# --- the real isolated run --------------------------------------------------
async def test_run_subagent_returns_answer_with_isolated_memory(
    mock_llm, tool_registry
):
    orch = Orchestrator(llm=mock_llm, tools_registry=tool_registry)
    answer = await orch.run_subagent("say hi", system_prompt="be terse", max_steps=3)
    assert isinstance(answer, str) and answer
    assert orch.shared_memory is None  # sub-agent used its own memory
