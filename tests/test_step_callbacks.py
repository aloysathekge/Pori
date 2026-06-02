"""Tests for the on_step_start / on_step_end lifecycle callbacks.

These callbacks let observers (SSE streaming in Pori Cloud, the CLI, custom
monitors) receive real per-step events instead of polling agent state. The
hooks were previously declared on Orchestrator.execute_task() but never wired
into Agent.run() — these tests lock in the wiring.
"""

import pytest

pytestmark = [pytest.mark.agent, pytest.mark.orchestrator]


async def test_run_invokes_sync_step_callbacks(test_agent_with_tool_calls):
    """Sync callbacks fire once per executed step and receive the agent."""
    agent = test_agent_with_tool_calls
    starts: list[int] = []
    ends: list[int] = []

    def on_start(a):
        # n_steps has not yet been incremented for the step about to run
        starts.append(a.state.n_steps)

    def on_end(a):
        ends.append(a.state.n_steps)

    await agent.run(on_step_start=on_start, on_step_end=on_end)

    # The tool-call mock runs test_tool -> answer -> done across steps.
    assert len(starts) >= 1
    assert len(starts) == len(ends)
    # on_step_end always observes at least as many completed steps as on_step_start
    for s, e in zip(starts, ends):
        assert e >= s


async def test_run_invokes_async_step_callbacks(test_agent_with_tool_calls):
    """Async callbacks are awaited, not left as un-awaited coroutines."""
    agent = test_agent_with_tool_calls
    seen: list[str] = []

    async def on_start(a):
        seen.append("start")

    async def on_end(a):
        seen.append("end")

    await agent.run(on_step_start=on_start, on_step_end=on_end)

    assert "start" in seen
    assert "end" in seen
    assert seen.count("start") == seen.count("end")


async def test_callback_errors_do_not_crash_run(test_agent_with_tool_calls):
    """A throwing callback is swallowed; the run still completes normally."""
    agent = test_agent_with_tool_calls

    def boom(a):
        raise RuntimeError("observer blew up")

    result = await agent.run(on_step_start=boom, on_step_end=boom)

    # Run reaches a terminal result rather than propagating the callback error.
    assert result["steps_taken"] >= 1
    assert "completed" in result


async def test_run_without_callbacks_is_backward_compatible(test_agent_with_tool_calls):
    """Calling run() with no callbacks keeps the original behavior."""
    result = await test_agent_with_tool_calls.run()
    assert result["completed"] is True


async def test_orchestrator_forwards_step_callbacks(orchestrator_with_tool_calls):
    """Orchestrator.execute_task forwards callbacks down to Agent.run()."""
    orch = orchestrator_with_tool_calls
    starts: list[object] = []
    ends: list[object] = []

    await orch.execute_task(
        task="Test task with tool calls",
        on_step_start=lambda a: starts.append(a),
        on_step_end=lambda a: ends.append(a),
    )

    assert len(starts) >= 1
    assert len(ends) >= 1
    # Callback receives the live Agent instance
    from pori.agent import Agent

    assert all(isinstance(a, Agent) for a in starts)
