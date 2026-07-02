"""AC-1 prompt caching helpers (pori/llm/prompt_caching.py)."""

import pytest

from pori.llm.prompt_caching import CACHE_CONTROL, cached_system

pytestmark = [pytest.mark.unit]


def test_cached_system_wraps_prompt_as_marked_block():
    out = cached_system("you are pori")
    assert out == [
        {
            "type": "text",
            "text": "you are pori",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_cached_system_empty_returns_empty_list():
    assert cached_system("") == []
    assert cached_system(None) == []  # type: ignore[arg-type]


def test_cached_system_marker_is_not_shared_reference():
    """Each block gets its own cache_control dict — never the module constant."""
    a = cached_system("a")[0]["cache_control"]
    b = cached_system("b")[0]["cache_control"]
    assert a == b == {"type": "ephemeral"}
    assert a is not b
    assert a is not CACHE_CONTROL
