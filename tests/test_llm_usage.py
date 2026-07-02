"""Normalized LLM usage (AC-4): pori/llm/messages.normalize_usage."""

import pytest

from pori.llm import Usage, normalize_usage

pytestmark = [pytest.mark.unit]


def test_normalize_anthropic_shape():
    u = normalize_usage(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 10,
        }
    )
    assert (u.input_tokens, u.output_tokens, u.total_tokens) == (100, 20, 120)
    assert (u.cache_read_tokens, u.cache_write_tokens) == (80, 10)


def test_normalize_openai_google_shape():
    u = normalize_usage(
        {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35}
    )
    assert (u.input_tokens, u.output_tokens, u.total_tokens) == (30, 5, 35)
    assert (u.cache_read_tokens, u.cache_write_tokens) == (0, 0)


def test_normalize_total_falls_back_to_sum_when_absent():
    u = normalize_usage({"input_tokens": 7, "output_tokens": 3})
    assert u.total_tokens == 10


def test_normalize_handles_none_and_garbage():
    assert normalize_usage(None) == Usage()
    assert normalize_usage("nope") == Usage()  # type: ignore[arg-type]
    # non-int values degrade to 0 rather than raising
    assert (
        normalize_usage({"input_tokens": None, "prompt_tokens": "x"}).input_tokens == 0
    )
