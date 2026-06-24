"""Tests for structured LLM output recovery behavior."""

import pytest
from pydantic import BaseModel

from pori.llm.openai import StructuredOutputParseError, StructuredWrapper


class TinyOutput(BaseModel):
    value: str


class BrokenStructuredLLM:
    async def ainvoke(self, messages, output_format=None):
        raise StructuredOutputParseError(
            "bad structured response",
            raw_content='{"value": "truncated',
        )


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
