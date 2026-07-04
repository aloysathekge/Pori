"""Background (async) delegation: registry, dispatcher, and the tool's bg path."""

import time

import pytest

from pori.background_delegation import BackgroundDelegationRegistry
from pori.subagents import make_background_delegate
from pori.tools.standard.core_tools import (
    DelegateTaskItem,
    DelegateTaskParams,
    delegate_task_tool,
)

pytestmark = [pytest.mark.unit]


def _drain_until(registry, n, timeout=2.0):
    deadline = time.monotonic() + timeout
    done = []
    while time.monotonic() < deadline:
        done.extend(registry.drain_completed())
        if len(done) >= n:
            break
        time.sleep(0.02)
    return done


def test_registry_dispatch_completes_and_drains():
    reg = BackgroundDelegationRegistry()

    async def work():
        return "hello"

    handle = reg.dispatch("g1", lambda: work())
    assert handle.startswith("bg-")
    done = _drain_until(reg, 1)
    assert len(done) == 1
    assert done[0].success and done[0].result == "hello" and done[0].goal == "g1"
    assert reg.active_count() == 0
    assert reg.drain_completed() == []  # queue cleared after draining


def test_registry_captures_child_error():
    reg = BackgroundDelegationRegistry()

    async def boom():
        raise ValueError("nope")

    reg.dispatch("g", lambda: boom())
    done = _drain_until(reg, 1)
    assert done[0].success is False and "nope" in done[0].error


def test_background_delegate_dispatches_and_results_drain():
    reg = BackgroundDelegationRegistry()

    class _Stub:
        async def run_subagent(self, goal, **kw):
            return f"done:{goal}"

    dispatch = make_background_delegate(_Stub(), reg)
    out = dispatch([{"goal": "a"}, {"goal": "b"}])
    assert all(o["success"] and o["handle"].startswith("bg-") for o in out)
    done = _drain_until(reg, 2)
    assert {d.result for d in done} == {"done:a", "done:b"}


def test_background_delegate_empty_goal_is_error():
    reg = BackgroundDelegationRegistry()
    out = make_background_delegate(object(), reg)([{"goal": "   "}])
    assert out[0]["success"] is False


def test_delegate_task_tool_background_dispatches():
    captured = {}

    def bg(items):
        captured["items"] = items
        return [{"success": True, "handle": "bg-9", "goal": items[0]["goal"]}]

    res = delegate_task_tool(
        DelegateTaskParams(tasks=[DelegateTaskItem(goal="a")], background=True),
        {"background_delegate": bg},
    )
    assert res["success"] is True and res["background"] is True
    assert res["dispatched"][0]["handle"] == "bg-9"
    assert captured["items"] == [{"goal": "a", "context": None, "agent": None}]


def test_delegate_task_tool_background_unavailable():
    res = delegate_task_tool(
        DelegateTaskParams(tasks=[DelegateTaskItem(goal="a")], background=True), {}
    )
    assert res["success"] is False and "background" in res["error"].lower()
