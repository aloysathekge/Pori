"""Aux-LLM context compression (AC-3): pori/compression.py."""

import pytest

from pori.compression import CompressionSummary, compress_context, render_summary
from pori.memory import AgentMemory, create_memory_store

pytestmark = [pytest.mark.unit]


class _FakeStructured:
    def __init__(self, summary, calls):
        self._summary = summary
        self._calls = calls

    async def ainvoke(self, messages):
        self._calls.append(1)
        return self._summary


class _FakeLLM:
    """Minimal LLM exposing with_structured_output, recording aux calls."""

    def __init__(self, summary):
        self._summary = summary
        self.calls: list = []

    def with_structured_output(self, model):
        return _FakeStructured(self._summary, self.calls)


def _memory_with_messages(n=12, size=200):
    mem = AgentMemory(store=create_memory_store(backend="memory"))
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        mem.add_message(role, f"message {i} " + "x" * size)
    return mem


async def test_compress_context_caches_llm_summary_and_window_uses_it():
    summary = CompressionSummary(
        active_task="do X", progress="did Y", key_facts=["a", "b"]
    )
    llm = _FakeLLM(summary)
    mem = _memory_with_messages()

    dropped, dropped_ids = mem.context_window_dropped(300, 100)
    assert dropped_ids  # the small budget drops older messages

    assert await compress_context(mem, llm, max_tokens=300, reserve_tokens=100) is True
    assert len(llm.calls) == 1
    assert mem.has_cached_summary(dropped_ids)

    # The sync window build now surfaces the LLM summary as the leading system msg
    window = mem.get_token_limited_messages(max_tokens=300, reserve_tokens=100)
    assert window[0]["role"] == "system"
    assert window[0]["content"] == render_summary(summary)
    assert "do X" in window[0]["content"]


async def test_compress_context_anti_thrash_reuses_cache():
    llm = _FakeLLM(CompressionSummary(active_task="t"))
    mem = _memory_with_messages()
    assert await compress_context(mem, llm, max_tokens=300, reserve_tokens=100) is True
    # Same dropped set → reused, not re-summarized.
    assert await compress_context(mem, llm, max_tokens=300, reserve_tokens=100) is False
    assert len(llm.calls) == 1


async def test_compress_context_fail_open_without_structured_output():
    class _Bare:
        pass

    mem = _memory_with_messages()
    assert (
        await compress_context(mem, _Bare(), max_tokens=300, reserve_tokens=100)
        is False
    )


async def test_compress_context_noop_when_nothing_dropped():
    llm = _FakeLLM(CompressionSummary(active_task="t"))
    mem = _memory_with_messages(n=2, size=10)  # tiny; fits the budget
    assert (
        await compress_context(mem, llm, max_tokens=3000, reserve_tokens=100) is False
    )
    assert llm.calls == []
