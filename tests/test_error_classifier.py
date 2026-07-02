"""LLM error classification + agent recovery (AC-2)."""

import asyncio

import pytest

from pori.agent import Agent, FatalAgentError
from pori.llm.error_classifier import FailoverReason, classify_error
from pori.llm.messages import ToolTurn
from pori.llm.retry import is_transient_error
from pori.memory import AgentMemory, create_memory_store

pytestmark = [pytest.mark.unit]


class _Status(Exception):
    def __init__(self, status, msg=""):
        super().__init__(msg or f"status {status}")
        self.status_code = status


def test_auth_and_billing_fail_fast_not_retryable():
    for status in (401, 403):
        c = classify_error(_Status(status))
        assert c.reason == FailoverReason.AUTH
        assert c.should_fail_fast and not c.retryable
    c = classify_error(_Status(402))
    assert c.reason == FailoverReason.BILLING
    assert c.should_fail_fast and not c.retryable


def test_rate_limit_and_server_and_timeout_are_retryable():
    assert classify_error(_Status(429)).reason == FailoverReason.RATE_LIMIT
    assert classify_error(_Status(503)).reason == FailoverReason.SERVER_ERROR
    assert classify_error(asyncio.TimeoutError()).reason == FailoverReason.TIMEOUT
    for exc in (_Status(429), _Status(503), asyncio.TimeoutError(), ConnectionError()):
        assert classify_error(exc).retryable is True


def test_context_overflow_triggers_compress_not_retry():
    c = classify_error(_Status(400, "prompt is too long: maximum context length"))
    assert c.reason == FailoverReason.CONTEXT_OVERFLOW
    assert c.should_compress and not c.retryable and not c.should_fail_fast
    # overflow detected even without a status code (bare BadRequest)
    assert classify_error(ValueError("this exceeds the context length")).should_compress


def test_generic_400_and_unknown_are_not_retryable():
    assert classify_error(_Status(400, "bad request")).reason == FailoverReason.UNKNOWN
    assert classify_error(ValueError("bad schema")).reason == FailoverReason.UNKNOWN
    assert classify_error(_Status(400)).retryable is False


def test_content_policy_classified_and_not_retryable():
    c = classify_error(_Status(400, "blocked by content policy"))
    assert c.reason == FailoverReason.CONTENT_POLICY_BLOCKED
    assert not c.retryable and not c.should_compress


def test_is_transient_error_delegates_to_classifier():
    assert is_transient_error(_Status(429)) is True
    assert is_transient_error(_Status(400)) is False
    assert is_transient_error(_Status(401)) is False


class _FailLLM:
    model = "m"

    def __init__(self, exc):
        self._exc = exc

    async def ainvoke_tools(self, messages, tools, on_event=None):
        raise self._exc


async def test_get_next_action_raises_fatal_on_auth(tool_registry, agent_settings):
    agent = Agent(
        task="t",
        llm=_FailLLM(_Status(401, "unauthorized")),
        tools_registry=tool_registry,
        settings=agent_settings,
        memory=AgentMemory(store=create_memory_store(backend="memory")),
    )
    with pytest.raises(FatalAgentError):
        await agent.get_next_action()


class _OverflowThenOK:
    model = "m"

    def __init__(self):
        self.calls = 0

    def with_structured_output(self, model):
        from pori.compression import CompressionSummary

        class _S:
            async def ainvoke(self, messages):
                return CompressionSummary(active_task="t")

        return _S()

    async def ainvoke_tools(self, messages, tools, on_event=None):
        self.calls += 1
        if self.calls == 1:
            raise _Status(400, "prompt is too long")
        return ToolTurn(text="done", tool_calls=[])


async def test_get_next_action_compresses_and_retries_on_overflow(
    tool_registry, agent_settings
):
    llm = _OverflowThenOK()
    agent = Agent(
        task="t",
        llm=llm,
        tools_registry=tool_registry,
        settings=agent_settings,
        memory=AgentMemory(store=create_memory_store(backend="memory")),
    )
    out = await agent.get_next_action()
    assert llm.calls == 2  # retried once after (attempted) compression
    assert out.action and "answer" in out.action[0]
