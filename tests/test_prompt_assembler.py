"""Tests for the cache-tiered system-prompt assembler."""

import pytest

from pori.agent import Agent, AgentSettings
from pori.memory import AgentMemory
from pori.prompts import DEFAULT_IDENTITY, SystemPromptTiers, build_system_prompt
from pori.tools.registry import ToolRegistry
from pori.tools.standard import register_all_tools

# --- assembler unit ---------------------------------------------------------


def test_tiers_render_stable_context_volatile_in_order():
    tiers = SystemPromptTiers()
    tiers.stable.append("IDENTITY")
    tiers.context.append("CUSTOM")
    tiers.volatile.append("SKILLS")
    out = build_system_prompt(tiers)
    assert out == "IDENTITY\n\nCUSTOM\n\nSKILLS"
    assert out.index("IDENTITY") < out.index("CUSTOM") < out.index("SKILLS")


def test_empty_and_whitespace_blocks_are_skipped():
    tiers = SystemPromptTiers()
    tiers.stable.append("A")
    tiers.context.append("")
    tiers.volatile.append("   ")
    assert build_system_prompt(tiers) == "A"


def test_default_identity_is_pori_and_neutral():
    assert "Pori" in DEFAULT_IDENTITY
    assert "open-source" in DEFAULT_IDENTITY


# --- agent integration ------------------------------------------------------


@pytest.fixture
def registry():
    r = ToolRegistry()
    register_all_tools(r)
    return r


class _StubLLM:
    def with_structured_output(self, model, include_raw=True):
        return self

    async def ainvoke(self, messages):  # pragma: no cover - not called in init
        raise AssertionError("LLM should not be invoked during prompt assembly")


def test_agent_prompt_has_identity_workflow_and_tools(registry):
    agent = Agent(
        task="t",
        llm=_StubLLM(),
        tools_registry=registry,
        settings=AgentSettings(max_steps=2),
        memory=AgentMemory(),
    )
    sm = agent.system_message
    # Stable identity comes first.
    assert sm.startswith("You are Pori")
    # Operating rules + tool guidance are present.
    assert "Workflow" in sm
    assert "answer" in sm  # a tool name from the injected descriptions


def test_agent_prompt_tier_order_identity_before_custom(registry):
    agent = Agent(
        task="t",
        llm=_StubLLM(),
        tools_registry=registry,
        settings=AgentSettings(max_steps=2),
        memory=AgentMemory(),
        system_prompt="CUSTOM_RULE_XYZ",
    )
    sm = agent.system_message
    # Identity/core (stable) precedes the caller's custom prompt (context).
    assert sm.index("You are Pori") < sm.index("CUSTOM_RULE_XYZ")
    # Identity is not duplicated by the old agent_core.md line.
    assert sm.count("You are Pori") == 1
