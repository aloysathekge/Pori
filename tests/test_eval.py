"""Tests for the evaluation and guardrail framework.

Covers ReliabilityEval (deterministic), AgentJudgeEval (binary + numeric
scoring, pre/post_check hooks), and the built-in guardrail subclasses.
"""

from dataclasses import dataclass
from typing import Any, List

import pytest

from pori.eval import (
    AgentJudgeEval,
    ContentPolicyGuardrail,
    FactualityGuardrail,
    ReliabilityEval,
    TopicGuardrail,
)
from pori.eval.agent_judge import BinaryJudgment, NumericJudgment

# ---- Mocks -----------------------------------------------------------------


class _StubToolCall:
    def __init__(self, name: str):
        self.tool_name = name


@dataclass
class _StubMemory:
    tool_call_history: List[_StubToolCall]


class _StubAgent:
    """Minimal agent stand-in for ReliabilityEval. `run()` is a no-op and the
    pre-seeded tool_call_history drives the result."""

    def __init__(self, tool_names: List[str]):
        self.memory = _StubMemory(
            tool_call_history=[_StubToolCall(n) for n in tool_names]
        )

    async def run(self):
        return None


class _MockJudgeLLM:
    """Judge LLM that returns a scripted sequence of BinaryJudgment /
    NumericJudgment objects. Matches the contract AgentJudgeEval expects:
    `with_structured_output(...).ainvoke(...)` returns a dict with a 'parsed'
    key holding the Pydantic model."""

    def __init__(self, judgments: List[Any]):
        self._judgments = list(judgments)
        self.call_count = 0

    def with_structured_output(self, schema, include_raw=True):
        self._schema = schema
        return self

    async def ainvoke(self, messages):
        if not self._judgments:
            raise AssertionError("MockJudgeLLM ran out of scripted judgments")
        self.call_count += 1
        return {"parsed": self._judgments.pop(0)}


# ---- ReliabilityEval -------------------------------------------------------


async def test_reliability_passes_when_all_expected_tools_called():
    agent = _StubAgent(["web_search", "answer", "done"])
    result = await ReliabilityEval(
        agent=agent, expected_tool_calls=["web_search", "answer"]
    ).run()

    assert result.passed is True
    assert result.failed_tool_calls == []
    assert set(result.passed_tool_calls) == {"web_search", "answer"}
    assert result.eval_type == "reliability"


async def test_reliability_fails_when_expected_tool_missing():
    agent = _StubAgent(["web_search"])  # "answer" missing
    result = await ReliabilityEval(
        agent=agent, expected_tool_calls=["web_search", "answer"]
    ).run()

    assert result.passed is False
    assert result.passed_tool_calls == ["web_search"]
    assert result.failed_tool_calls == ["answer"]


async def test_reliability_passes_trivially_when_no_expectations():
    agent = _StubAgent(["whatever"])
    result = await ReliabilityEval(agent=agent, expected_tool_calls=[]).run()
    assert result.passed is True


async def test_reliability_ignores_extra_unexpected_tools():
    """Extra tool calls beyond the expected set don't cause failure — the
    check is 'all expected were called', not 'only expected were called'."""
    agent = _StubAgent(["web_search", "bonus_tool", "answer"])
    result = await ReliabilityEval(
        agent=agent, expected_tool_calls=["web_search", "answer"]
    ).run()
    assert result.passed is True


# ---- AgentJudgeEval (binary) ----------------------------------------------


async def test_agent_judge_binary_pass():
    judge = _MockJudgeLLM([BinaryJudgment(passed=True, reason="looks good")])
    eval_ = AgentJudgeEval(criteria="Be concise", judge_llm=judge, scoring="binary")
    result = await eval_.run(input="q", output="a")

    assert result.passed is True
    assert result.pass_rate == 1.0
    assert result.reason == "looks good"
    assert len(result.evaluations) == 1


async def test_agent_judge_binary_fail():
    judge = _MockJudgeLLM([BinaryJudgment(passed=False, reason="too verbose")])
    eval_ = AgentJudgeEval(criteria="Be concise", judge_llm=judge, scoring="binary")
    result = await eval_.run(input="q", output="a")

    assert result.passed is False
    assert result.pass_rate == 0.0


async def test_agent_judge_on_fail_callback_invoked():
    captured: List[dict] = []
    judge = _MockJudgeLLM([BinaryJudgment(passed=False, reason="nope")])
    eval_ = AgentJudgeEval(
        criteria="x",
        judge_llm=judge,
        scoring="binary",
        on_fail=captured.append,
    )
    await eval_.run(input="i", output="o")

    assert len(captured) == 1
    assert captured[0]["passed"] is False
    assert captured[0]["reason"] == "nope"


async def test_agent_judge_on_fail_not_invoked_on_pass():
    captured: List[dict] = []
    judge = _MockJudgeLLM([BinaryJudgment(passed=True, reason="ok")])
    eval_ = AgentJudgeEval(
        criteria="x",
        judge_llm=judge,
        scoring="binary",
        on_fail=captured.append,
    )
    await eval_.run(input="i", output="o")

    assert captured == []


# ---- AgentJudgeEval (numeric) ---------------------------------------------


async def test_agent_judge_numeric_meets_threshold():
    judge = _MockJudgeLLM([NumericJudgment(score=8, reason="solid")])
    eval_ = AgentJudgeEval(
        criteria="x", judge_llm=judge, scoring="numeric", threshold=7
    )
    result = await eval_.run(input="i", output="o")

    assert result.passed is True
    assert result.score == 8
    assert result.evaluations[0]["score"] == 8


async def test_agent_judge_numeric_below_threshold():
    judge = _MockJudgeLLM([NumericJudgment(score=5, reason="meh")])
    eval_ = AgentJudgeEval(
        criteria="x", judge_llm=judge, scoring="numeric", threshold=7
    )
    result = await eval_.run(input="i", output="o")

    assert result.passed is False
    assert result.score == 5


# ---- AgentJudgeEval multi-case --------------------------------------------


async def test_agent_judge_computes_pass_rate_over_multiple_cases():
    judge = _MockJudgeLLM(
        [
            BinaryJudgment(passed=True, reason="a"),
            BinaryJudgment(passed=False, reason="b"),
            BinaryJudgment(passed=True, reason="c"),
        ]
    )
    eval_ = AgentJudgeEval(criteria="x", judge_llm=judge, scoring="binary")
    result = await eval_.run(
        cases=[
            {"input": "i1", "output": "o1"},
            {"input": "i2", "output": "o2"},
            {"input": "i3", "output": "o3"},
        ]
    )

    assert len(result.evaluations) == 3
    assert result.pass_rate == pytest.approx(2 / 3)
    # passed=True only when pass_rate == 1.0
    assert result.passed is False


# ---- Guardrail hooks (pre_check / post_check) -----------------------------


async def test_pre_check_raises_on_fail():
    judge = _MockJudgeLLM([BinaryJudgment(passed=False, reason="blocked input")])
    eval_ = AgentJudgeEval(criteria="x", judge_llm=judge, scoring="binary")

    with pytest.raises(ValueError, match="Input guardrail failed"):
        await eval_.pre_check("user input")


async def test_pre_check_silent_on_pass():
    judge = _MockJudgeLLM([BinaryJudgment(passed=True, reason="ok")])
    eval_ = AgentJudgeEval(criteria="x", judge_llm=judge, scoring="binary")
    # should not raise
    await eval_.pre_check("user input")


async def test_post_check_raises_on_fail():
    judge = _MockJudgeLLM([BinaryJudgment(passed=False, reason="bad output")])
    eval_ = AgentJudgeEval(criteria="x", judge_llm=judge, scoring="binary")

    with pytest.raises(ValueError, match="Output guardrail failed"):
        await eval_.post_check("user input", "agent output")


# ---- Built-in guardrail subclasses ----------------------------------------


def test_content_policy_guardrail_wires_binary_scoring():
    guard = ContentPolicyGuardrail(judge_llm=_MockJudgeLLM([]))
    assert guard.scoring == "binary"
    assert guard.name == "content_policy"
    assert "hate speech" in guard.criteria


def test_factuality_guardrail_wires_binary_scoring():
    guard = FactualityGuardrail(judge_llm=_MockJudgeLLM([]))
    assert guard.scoring == "binary"
    assert guard.name == "factuality"
    assert "factual" in guard.criteria


def test_topic_guardrail_interpolates_allowed_topics():
    guard = TopicGuardrail(
        allowed_topics=["science", "technology"], judge_llm=_MockJudgeLLM([])
    )
    assert guard.scoring == "binary"
    assert guard.name == "topic_restriction"
    assert "science, technology" in guard.criteria


async def test_topic_guardrail_post_check_blocks_off_topic():
    """End-to-end: the subclass's inherited post_check should raise when the
    judge says fail."""
    judge = _MockJudgeLLM([BinaryJudgment(passed=False, reason="off-topic")])
    guard = TopicGuardrail(allowed_topics=["science"], judge_llm=judge)

    with pytest.raises(ValueError, match="Output guardrail failed"):
        await guard.post_check("tell me a joke", "here is a joke")
