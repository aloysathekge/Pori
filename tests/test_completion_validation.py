"""Tests for completion-quality gating on the 'answer' tool.

Two layers:
  1. Deterministic (always on): an empty/whitespace final_answer is rejected.
  2. Semantic (opt-in via settings.validate_output): an LLM judge rejects
     inadequate answers, bounded by settings.max_validation_retries.
"""

import pytest

from pori.agent import Agent, AgentSettings, CompletionValidation

pytestmark = pytest.mark.agent


class _ValidatorLLM:
    """Mock LLM whose structured-output call yields CompletionValidation verdicts.

    `verdicts` is a sequence of bool (adequate) or Exception (to simulate a
    validator outage). The last entry repeats once exhausted.
    """

    model = "mock-validator"

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self._i = 0
        self.calls = 0

    def with_structured_output(self, output_model, include_raw=False):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self._i < len(self._verdicts):
            v = self._verdicts[self._i]
            self._i += 1
        else:
            v = self._verdicts[-1] if self._verdicts else True
        if isinstance(v, Exception):
            raise v
        return CompletionValidation(
            adequate=bool(v), reason="" if v else "core request unaddressed"
        )


def _make_agent(llm, tool_registry, memory, *, validate_output=False, cap=2):
    return Agent(
        task="What is the capital of France?",
        llm=llm,
        tools_registry=tool_registry,
        settings=AgentSettings(
            max_steps=5,
            max_failures=2,
            validate_output=validate_output,
            max_validation_retries=cap,
        ),
        memory=memory,
    )


async def test_empty_answer_rejected(mock_llm, tool_registry, legacy_memory):
    agent = _make_agent(mock_llm, tool_registry, legacy_memory)
    results = await agent.execute_actions(
        [{"answer": {"final_answer": "", "reasoning": "none"}}]
    )
    assert results[0].success is False
    assert legacy_memory.get_final_answer() is None


async def test_whitespace_answer_rejected(mock_llm, tool_registry, legacy_memory):
    agent = _make_agent(mock_llm, tool_registry, legacy_memory)
    results = await agent.execute_actions(
        [{"answer": {"final_answer": "   \n  ", "reasoning": "none"}}]
    )
    assert results[0].success is False
    assert legacy_memory.get_final_answer() is None


async def test_nonempty_answer_accepted_without_validation(
    mock_llm, tool_registry, legacy_memory
):
    agent = _make_agent(mock_llm, tool_registry, legacy_memory)
    results = await agent.execute_actions(
        [{"answer": {"final_answer": "Paris.", "reasoning": "geography"}}]
    )
    assert results[0].success is True
    assert legacy_memory.get_final_answer()["final_answer"] == "Paris."


async def test_semantic_validation_rejects_inadequate(tool_registry, legacy_memory):
    llm = _ValidatorLLM([False])
    agent = _make_agent(llm, tool_registry, legacy_memory, validate_output=True)
    results = await agent.execute_actions(
        [{"answer": {"final_answer": "I don't know.", "reasoning": "x"}}]
    )
    assert results[0].success is False
    assert "validation" in (results[0].error or "").lower()
    assert legacy_memory.get_final_answer() is None
    assert agent._completion_validation_attempts == 1
    assert llm.calls == 1


async def test_semantic_validation_accepts_adequate(tool_registry, legacy_memory):
    llm = _ValidatorLLM([True])
    agent = _make_agent(llm, tool_registry, legacy_memory, validate_output=True)
    results = await agent.execute_actions(
        [{"answer": {"final_answer": "Paris is the capital.", "reasoning": "x"}}]
    )
    assert results[0].success is True
    assert legacy_memory.get_final_answer()["final_answer"] == "Paris is the capital."


async def test_validation_retry_cap_eventually_accepts(tool_registry, legacy_memory):
    # Judge always says inadequate; after `cap` rejections the gate is skipped
    # and the answer is accepted so the agent doesn't burn all its steps.
    llm = _ValidatorLLM([False])
    agent = _make_agent(llm, tool_registry, legacy_memory, validate_output=True, cap=2)
    action = [{"answer": {"final_answer": "Maybe Paris?", "reasoning": "x"}}]

    r1 = await agent.execute_actions(action)
    r2 = await agent.execute_actions(action)
    r3 = await agent.execute_actions(action)

    assert r1[0].success is False
    assert r2[0].success is False
    assert r3[0].success is True  # cap reached -> accepted
    assert legacy_memory.get_final_answer()["final_answer"] == "Maybe Paris?"
    assert llm.calls == 2  # validator only consulted up to the cap


async def test_validation_fails_open_on_validator_error(tool_registry, legacy_memory):
    # If the validator itself errors, the answer is accepted (fail-open).
    llm = _ValidatorLLM([RuntimeError("validator down")])
    agent = _make_agent(llm, tool_registry, legacy_memory, validate_output=True)
    results = await agent.execute_actions(
        [{"answer": {"final_answer": "Paris.", "reasoning": "x"}}]
    )
    assert results[0].success is True
    assert legacy_memory.get_final_answer()["final_answer"] == "Paris."
