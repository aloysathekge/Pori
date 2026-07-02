"""Anthropic prompt caching (Pori AC-1; ported from Hermes' ``prompt_caching.py``).

Pure functions that add Anthropic ``cache_control`` breakpoints so the provider
reuses a stable prefix across turns instead of re-billing it on every step.

Anthropic caches in prefix order: **tools, then system, then messages**. A
breakpoint on the *system* block therefore caches the entire ``tools + system``
prefix — the large, stable part of Pori's request, and the bulk of the savings.
It is applied on every call; the system prompt is built once per run
(``Agent._setup_system_message``), so the cached prefix stays warm across steps.
The cache-hit accounting already exists (``ChatAnthropic`` reads
``cache_read_input_tokens`` / ``cache_creation_input_tokens``); it was simply
always ~0 because nothing set a breakpoint.

A sliding window over the last *N* messages is intentionally NOT applied yet:
Pori currently appends volatile per-step context at the tail of the message
list, so marking recent messages would write a fresh cache entry each step with
no read benefit. That lands with the ``_build_messages`` restructure (AC-1b),
which moves the volatile context into a single trailing message.
"""

from typing import Any

# Anthropic ephemeral cache marker (default ~5-minute TTL). A fresh dict is
# handed out per use so callers can never alias/mutate a shared marker.
CACHE_CONTROL: dict[str, str] = {"type": "ephemeral"}


def cached_system(system_prompt: str) -> list[dict[str, Any]]:
    """Return the system prompt as a cache-marked Anthropic text block.

    Marking the system block caches the ``tools + system`` prefix (tools precede
    system in Anthropic's cache order). Returns ``[]`` for an empty prompt.
    """
    if not system_prompt:
        return []
    return [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": dict(CACHE_CONTROL),
        }
    ]
