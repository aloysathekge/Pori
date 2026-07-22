"""Tests for structured LLM output recovery behavior."""

import pytest
from pydantic import BaseModel

from pori.llm import UserMessage
from pori.llm.openai import ChatOpenAI, StructuredOutputParseError, StructuredWrapper


class TinyOutput(BaseModel):
    value: str


class BrokenStructuredLLM:
    async def ainvoke(self, messages, output_format=None):
        raise StructuredOutputParseError(
            "bad structured response",
            raw_content='{"value": "truncated',
        )


class _StreamChunk:
    def __init__(self, content=None, usage=None):
        self.choices = (
            [type("Choice", (), {"delta": type("Delta", (), {"content": content})()})()]
            if content is not None
            else []
        )
        self.usage = usage


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def generate():
            for chunk in self._chunks:
                yield chunk

        return generate()


class _StreamCompletions:
    def __init__(self, chunks):
        self._chunks = chunks
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeStream(self._chunks)


class _StreamClient:
    def __init__(self, chunks):
        completions = _StreamCompletions(chunks)
        self.chat = type("Chat", (), {"completions": completions})()


@pytest.mark.asyncio
async def test_structured_wrapper_preserves_raw_parse_failure():
    wrapper = StructuredWrapper(
        BrokenStructuredLLM(),
        TinyOutput,
        include_raw=True,
    )

    result = await wrapper.ainvoke([])

    assert result["parsed"] is None
    assert result["raw"] == '{"value": "truncated'
    assert "bad structured response" in result["error"]


@pytest.mark.asyncio
async def test_structured_wrapper_raises_parse_failure_without_raw_mode():
    wrapper = StructuredWrapper(
        BrokenStructuredLLM(),
        TinyOutput,
        include_raw=False,
    )

    with pytest.raises(StructuredOutputParseError):
        await wrapper.ainvoke([])


@pytest.mark.asyncio
async def test_openai_structured_stream_reports_deltas_then_validates_once():
    usage = type(
        "Usage",
        (),
        {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
    )()
    client = _StreamClient(
        [
            _StreamChunk('{"value":'),
            _StreamChunk('"ready"}'),
            _StreamChunk(usage=usage),
        ]
    )
    llm = ChatOpenAI(api_key="x", model="structured-stream-test")
    llm._client = client
    deltas: list[str] = []

    result = await llm.with_structured_output(TinyOutput).ainvoke_with_deltas(
        [UserMessage(content="Return ready")],
        deltas.append,
    )

    assert result == TinyOutput(value="ready")
    assert deltas == ['{"value":', '"ready"}']
    sent = client.chat.completions.last_kwargs
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    assert sent["response_format"]["type"] == "json_schema"
    assert llm.last_usage == {
        "prompt_tokens": 4,
        "completion_tokens": 3,
        "total_tokens": 7,
    }
