"""OpenAI-compatible clients: session affinity and cached-token accounting.

Serverless prefix caches (Fireworks) are per-replica and report reused prompt
tokens under ``prompt_tokens_details.cached_tokens``. The client must pin a
logical session to one replica via the ``user`` field and surface cached
tokens so cost accounting can price them at the cached rate.
"""

from types import SimpleNamespace

import pytest

from pori.llm.messages import UserMessage, normalize_usage
from pori.llm.openai import ChatOpenAI


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_kwargs: dict = {}

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        usage = SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=20,
            total_tokens=1020,
            prompt_tokens_details=SimpleNamespace(cached_tokens=900),
        )
        message = SimpleNamespace(content="ok", tool_calls=[])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=usage,
        )


@pytest.fixture
def model() -> tuple[ChatOpenAI, _FakeCompletions]:
    llm = ChatOpenAI(model="test-model", api_key="test-key")
    completions = _FakeCompletions()
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return llm, completions


async def test_tool_calls_carry_session_affinity_and_cached_usage(model):
    llm, completions = model
    llm.session_affinity = "run-abc123"
    await llm.ainvoke_tools(
        [UserMessage(content="hello")],
        [{"name": "noop", "description": "", "input_schema": {"type": "object"}}],
    )
    assert completions.last_kwargs["user"] == "run-abc123"
    usage = normalize_usage(llm.last_usage)
    assert usage.input_tokens == 1000
    assert usage.cache_read_tokens == 900


async def test_plain_invoke_without_affinity_sends_no_user_field(model):
    llm, completions = model
    result = await llm.ainvoke([UserMessage(content="hello")])
    assert result == "ok"
    assert "user" not in completions.last_kwargs
    assert normalize_usage(llm.last_usage).cache_read_tokens == 900
