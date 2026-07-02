"""Duplicate-run guard in the Orchestrator (GW-3)."""

import asyncio

import pytest

from pori.orchestrator.core import ConversationBusy, Orchestrator

pytestmark = [pytest.mark.orchestrator]


async def test_reject_when_session_key_busy(mock_llm, tool_registry):
    orch = Orchestrator(llm=mock_llm, tools_registry=tool_registry)

    async def _pending():
        await asyncio.sleep(0.2)
        return {"ok": True}

    fut = asyncio.ensure_future(_pending())
    orch._active_sessions["key1"] = fut
    try:
        with pytest.raises(ConversationBusy):
            await orch.execute_task("task", session_key="key1", on_busy="reject")
    finally:
        fut.cancel()


async def test_coalesce_awaits_the_in_flight_run(mock_llm, tool_registry):
    orch = Orchestrator(llm=mock_llm, tools_registry=tool_registry)
    sentinel = {"coalesced": True}

    async def _pending():
        await asyncio.sleep(0.05)
        return sentinel

    orch._active_sessions["k"] = asyncio.ensure_future(_pending())
    result = await orch.execute_task("task", session_key="k", on_busy="coalesce")
    assert result is sentinel


async def test_slot_released_after_run(mock_llm, tool_registry):
    # A completed run must leave no lingering slot, so the next submit is allowed.
    orch = Orchestrator(llm=mock_llm, tools_registry=tool_registry)
    await orch.execute_task("say hi", session_key="lane-a")
    assert "lane-a" not in orch._active_sessions
