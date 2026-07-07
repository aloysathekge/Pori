"""Cross-provider failover chain (Hermes gap Tier 1.3)."""

import pytest

from pori.llm.failover import FailoverChatModel
from pori.llm.messages import ToolTurn, UserMessage


class RateLimitError(Exception):
    """Name-classified as RATE_LIMIT by the error classifier."""


class ContextOverflowError(Exception):
    def __init__(self):
        super().__init__("prompt is too long: maximum context length exceeded")


class FakeModel:
    def __init__(self, name, fail_with=None, fail_times=None):
        self.model = name
        self.max_tokens = 4096
        self.last_usage = None
        self.fail_with = fail_with
        self.fail_times = fail_times  # None = always fail (when fail_with set)
        self.calls = 0

    def _maybe_fail(self):
        self.calls += 1
        if self.fail_with is None:
            return
        if self.fail_times is None or self.calls <= self.fail_times:
            raise self.fail_with

    async def ainvoke(self, messages, output_format=None):
        self._maybe_fail()
        return f"answer from {self.model}"

    async def ainvoke_tools(self, messages, tools, on_event=None):
        self._maybe_fail()
        return ToolTurn(text=f"turn from {self.model}")


MSGS = [UserMessage(content="hi")]


class TestFailover:
    async def test_healthy_primary_never_touches_fallback(self):
        primary = FakeModel("primary")
        fallback = FakeModel("fallback")
        chain = FailoverChatModel([primary, fallback])

        assert await chain.ainvoke(MSGS) == "answer from primary"
        assert fallback.calls == 0
        assert chain.model == "primary"

    async def test_rate_limited_primary_fails_over(self):
        primary = FakeModel("primary", fail_with=RateLimitError("429"))
        fallback = FakeModel("fallback")
        chain = FailoverChatModel([primary, fallback])

        assert await chain.ainvoke(MSGS) == "answer from fallback"
        # Sticky: the next call starts at the survivor, not the dead primary
        assert await chain.ainvoke(MSGS) == "answer from fallback"
        assert primary.calls == 1
        assert chain.model == "fallback"

    async def test_tools_path_fails_over_too(self):
        primary = FakeModel("primary", fail_with=ConnectionError("unreachable"))
        fallback = FakeModel("fallback")
        chain = FailoverChatModel([primary, fallback])

        turn = await chain.ainvoke_tools(MSGS, tools=[])
        assert turn.text == "turn from fallback"

    async def test_context_overflow_is_not_a_failover(self):
        primary = FakeModel("primary", fail_with=ContextOverflowError())
        fallback = FakeModel("fallback")
        chain = FailoverChatModel([primary, fallback])

        with pytest.raises(ContextOverflowError):
            await chain.ainvoke(MSGS)
        assert fallback.calls == 0

    async def test_exhausted_chain_raises_last_error(self):
        primary = FakeModel("primary", fail_with=RateLimitError("429"))
        fallback = FakeModel("fallback", fail_with=RateLimitError("429 too"))
        chain = FailoverChatModel([primary, fallback])

        with pytest.raises(RateLimitError):
            await chain.ainvoke(MSGS)

    async def test_attribute_proxying_follows_active(self):
        primary = FakeModel("primary", fail_with=RateLimitError("429"))
        fallback = FakeModel("fallback")
        fallback.last_usage = {"input_tokens": 7}
        chain = FailoverChatModel([primary, fallback])

        await chain.ainvoke(MSGS)
        assert chain.last_usage == {"input_tokens": 7}
        assert chain.max_tokens == 4096

    async def test_empty_chain_rejected(self):
        with pytest.raises(ValueError):
            FailoverChatModel([])


class TestConfigWiring:
    def test_create_llm_without_fallbacks_returns_bare_model(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from pori.config import LLMConfig, create_llm

        llm = create_llm(LLMConfig(provider="anthropic", model="claude-sonnet-5"))
        assert type(llm).__name__ == "ChatAnthropic"

    def test_create_llm_with_fallbacks_builds_chain(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        from pori.config import LLMConfig, LLMFallback, create_llm

        llm = create_llm(
            LLMConfig(
                provider="anthropic",
                model="claude-sonnet-5",
                fallbacks=[LLMFallback(provider="openai", model="gpt-4o")],
            )
        )
        assert type(llm).__name__ == "FailoverChatModel"
        assert llm.model == "claude-sonnet-5"
        assert llm.active_label == "anthropic:claude-sonnet-5"

    def test_unavailable_fallback_skipped_not_fatal(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from pori.config import LLMConfig, LLMFallback, create_llm

        llm = create_llm(
            LLMConfig(
                provider="anthropic",
                model="claude-sonnet-5",
                fallbacks=[LLMFallback(provider="openai", model="gpt-4o")],
            )
        )
        # The only fallback was unbuildable -> bare primary, no chain
        assert type(llm).__name__ == "ChatAnthropic"
