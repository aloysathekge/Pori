"""Autonomous background review after a run (SK-1, layer 2)."""

import asyncio

import pytest

from pori.agent import AgentSettings
from pori.memory import AgentMemory
from pori.orchestrator.core import Orchestrator
from pori.skills_learn import build_background_review_prompt

pytestmark = [pytest.mark.orchestrator]


def test_background_review_prompt_is_conservative_and_actionable():
    prompt = build_background_review_prompt(
        "user: how to deploy\nassistant: run deploy.sh"
    )
    assert "deploy.sh" in prompt  # the digest is embedded
    assert "write_skill" in prompt
    assert "conservative" in prompt.lower()


async def test_spawn_creates_task_with_the_session_digest(
    mock_llm, tool_registry, monkeypatch
):
    orch = Orchestrator(llm=mock_llm, tools_registry=tool_registry)
    seen = {}

    async def fake_review(digest):
        seen["digest"] = digest

    monkeypatch.setattr(orch, "_run_background_review", fake_review)
    memory = AgentMemory()
    memory.add_message("user", "how to deploy the service")
    memory.add_message("assistant", "run ./deploy.sh in the repo root")

    orch._spawn_background_review(memory)
    assert orch._review_tasks  # a fire-and-forget task was registered
    await asyncio.gather(*list(orch._review_tasks), return_exceptions=True)
    assert "deploy" in seen["digest"]


def test_no_review_for_an_empty_session(mock_llm, tool_registry):
    orch = Orchestrator(llm=mock_llm, tools_registry=tool_registry)
    orch._spawn_background_review(AgentMemory())  # empty -> no digest, no task
    assert not orch._review_tasks


async def test_execute_task_triggers_review_only_when_enabled(
    mock_llm, tool_registry, monkeypatch
):
    orch = Orchestrator(llm=mock_llm, tools_registry=tool_registry)
    calls = {"n": 0}
    monkeypatch.setattr(
        orch,
        "_spawn_background_review",
        lambda memory: calls.__setitem__("n", calls["n"] + 1),
    )

    await orch.execute_task("x", agent_settings=AgentSettings(background_review=False))
    assert calls["n"] == 0

    await orch.execute_task("x", agent_settings=AgentSettings(background_review=True))
    assert calls["n"] == 1
