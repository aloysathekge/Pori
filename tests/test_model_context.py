"""Model-aware context sizing: pori/llm/model_context + Agent auto-sizing."""

import pytest

from pori.agent import Agent, AgentSettings
from pori.llm.model_context import DEFAULT_CONTEXT_LENGTH, get_model_context_length
from pori.memory import AgentMemory, create_memory_store

pytestmark = [pytest.mark.unit]


def test_known_model_context_lengths():
    assert get_model_context_length("claude-sonnet-4-20250514") == 200_000
    assert get_model_context_length("gpt-4o") == 128_000
    assert get_model_context_length("openai/gpt-4.1-mini") == 1_000_000
    assert get_model_context_length("gemini-2.5-flash") == 1_000_000
    assert get_model_context_length("o3-mini") == 200_000


def test_specific_markers_win_over_general():
    # gemini-1.5-pro (2M) must not be shadowed by the general gemini (1M) entry.
    assert get_model_context_length("gemini-1.5-pro-latest") == 2_000_000


def test_unknown_and_empty_fall_back_to_default():
    assert get_model_context_length("some-random-model") == DEFAULT_CONTEXT_LENGTH
    assert get_model_context_length("") == DEFAULT_CONTEXT_LENGTH
    assert get_model_context_length(None) == DEFAULT_CONTEXT_LENGTH  # type: ignore[arg-type]


class _LLM:
    max_tokens = 4096

    def __init__(self, model):
        self.model = model


def test_agent_auto_sizes_window_from_model(tool_registry):
    agent = Agent(
        task="t",
        llm=_LLM("claude-sonnet-4"),
        tools_registry=tool_registry,
        settings=AgentSettings(),  # auto on by default
        memory=AgentMemory(store=create_memory_store(backend="memory")),
    )
    assert agent.settings.context_window_tokens == 200_000  # not the fixed 3000
    assert agent.settings.context_window_reserve_tokens >= 4096 + 8000


def test_agent_respects_explicit_window_when_auto_off(tool_registry):
    agent = Agent(
        task="t",
        llm=_LLM("claude-sonnet-4"),
        tools_registry=tool_registry,
        settings=AgentSettings(context_window_auto=False, context_window_tokens=5000),
        memory=AgentMemory(store=create_memory_store(backend="memory")),
    )
    assert agent.settings.context_window_tokens == 5000
