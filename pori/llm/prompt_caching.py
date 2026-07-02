"""Anthropic prompt caching (Pori AC-1; ported from Hermes' ``prompt_caching.py``).

Pure functions that add Anthropic ``cache_control`` breakpoints so the provider
reuses a stable prefix across turns instead of re-billing it on every step.

Anthropic caches in prefix order: **tools, then system, then messages**, and
allows up to 4 breakpoints total.

- A breakpoint on the *system* block caches the entire ``tools + system`` prefix
  — the large, stable part of Pori's request, and the bulk of the savings
  (:func:`cached_system`, applied by the Anthropic wrapper on every call).
- Breakpoints on the last few *messages* extend the cache into recent history
  (:func:`mark_last_messages`). This only pays off when a single volatile
  message sits at the very tail: ``Agent._build_messages`` puts the per-step
  state (Runtime Facts, recent actions) LAST, so the stable messages just before
  it — system, history, frozen context, task — form a cacheable prefix and only
  the trailing message is re-billed each step.

The cache-hit accounting already exists (``ChatAnthropic`` reads
``cache_read_input_tokens`` / ``cache_creation_input_tokens``); it was simply
always ~0 before any breakpoint was set.
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


def mark_last_messages(messages: list[dict[str, Any]], n: int) -> None:
    """Add a cache breakpoint to each of the last ``n`` messages, in place.

    Anthropic allows 4 breakpoints total and one is spent on the system block,
    so ``n`` should be ``<= 3``. Marking is a no-op for messages whose content
    can't carry a marker (e.g. empty strings).
    """
    if n <= 0:
        return
    marked = 0
    for message in reversed(messages):
        if marked >= n:
            break
        if _apply_cache_marker(message):
            marked += 1


def _apply_cache_marker(message: dict[str, Any]) -> bool:
    """Mark a message's final content block with ``cache_control``.

    A string content is promoted to a single text block; a list content gets the
    marker on its last block. Returns ``True`` when a marker was applied.
    """
    content = message.get("content")
    if isinstance(content, str):
        if not content:
            return False
        message["content"] = [
            {"type": "text", "text": content, "cache_control": dict(CACHE_CONTROL)}
        ]
        return True
    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = dict(CACHE_CONTROL)
            return True
    return False
