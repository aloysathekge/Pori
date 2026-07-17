"""AC-1 prompt caching helpers (pori/llm/prompt_caching.py)."""

import pytest

from pori.llm import UserMessage
from pori.llm.prompt_caching import (
    CACHE_CONTROL,
    cached_system,
    mark_last_messages,
    mark_message_prefixes,
)

pytestmark = [pytest.mark.unit]

_EPHEMERAL = {"type": "ephemeral"}


def test_cached_system_wraps_prompt_as_marked_block():
    out = cached_system("you are pori")
    assert out == [
        {"type": "text", "text": "you are pori", "cache_control": _EPHEMERAL}
    ]


def test_cached_system_empty_returns_empty_list():
    assert cached_system("") == []
    assert cached_system(None) == []  # type: ignore[arg-type]


def test_cached_system_marker_is_not_shared_reference():
    """Each block gets its own cache_control dict — never the module constant."""
    a = cached_system("a")[0]["cache_control"]
    b = cached_system("b")[0]["cache_control"]
    assert a == b == _EPHEMERAL
    assert a is not b
    assert a is not CACHE_CONTROL


def test_mark_last_messages_marks_last_n_only():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    mark_last_messages(msgs, 2)
    assert msgs[0]["content"] == "a"  # untouched
    assert msgs[1]["content"] == [
        {"type": "text", "text": "b", "cache_control": _EPHEMERAL}
    ]
    assert msgs[2]["content"] == [
        {"type": "text", "text": "c", "cache_control": _EPHEMERAL}
    ]


def test_mark_last_messages_marks_last_block_of_list_content():
    msgs = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
        }
    ]
    mark_last_messages(msgs, 1)
    assert msgs[0]["content"][-1]["cache_control"] == _EPHEMERAL


def test_mark_last_messages_noop_for_zero_and_empty_content():
    msgs = [{"role": "user", "content": "x"}]
    mark_last_messages(msgs, 0)
    assert msgs[0]["content"] == "x"  # n=0 → untouched
    empty = [{"role": "user", "content": ""}]
    mark_last_messages(empty, 3)
    assert empty[0]["content"] == ""  # empty string can't carry a marker → skipped


def test_mark_message_prefixes_honors_host_breakpoint_then_marks_tail():
    source = [
        UserMessage(content="trusted", cache_breakpoint=True),
        UserMessage(content="history"),
        UserMessage(content="task"),
        UserMessage(content="volatile"),
    ]
    provider = [{"role": "user", "content": message.content} for message in source]

    mark_message_prefixes(source, provider, max_breakpoints=3)

    assert provider[0]["content"][0]["cache_control"] == _EPHEMERAL
    assert provider[1]["content"] == "history"
    assert provider[2]["content"][0]["cache_control"] == _EPHEMERAL
    assert provider[3]["content"][0]["cache_control"] == _EPHEMERAL


def test_mark_message_prefixes_disables_message_cache_for_restricted_context():
    source = [
        UserMessage(content="restricted", cache_breakpoint=False, cacheable=False),
        UserMessage(content="task"),
    ]
    provider = [{"role": "user", "content": message.content} for message in source]

    mark_message_prefixes(source, provider)

    assert provider == [
        {"role": "user", "content": "restricted"},
        {"role": "user", "content": "task"},
    ]
