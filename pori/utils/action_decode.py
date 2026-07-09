"""Decode a model-emitted JSON action envelope from assistant text.

Pori drives the agent loop with native tool-calling, but providers are not
uniform: a model sometimes emits its action decision as a JSON object of *text*
(``{"current_state": {...}, "action": [{"tool": {...}}]}``) instead of a real
tool call — Gemini does it when its native tool schema is rejected, and even
strong models do it intermittently. When that happens the agent must decode and
run those actions, not freeze the raw JSON as the final answer. This is the
decode contract: actions arrive as native tool-calls OR this envelope.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```$")


def _strip(text: str) -> str:
    t = text.strip()
    t = _FENCE_OPEN.sub("", t)
    t = _FENCE_CLOSE.sub("", t)
    return t.strip()


def looks_like_action_envelope(text: str) -> bool:
    """Cheap guard: text is a JSON object mentioning an ``action`` (don't stream
    it as prose)."""
    if not text:
        return False
    t = _strip(text)
    return t.startswith("{") and '"action"' in t


def decode_action_envelope(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse ``text`` into a list of ``{tool_name: args}`` actions.

    Returns None when the text isn't a well-formed action envelope, so ordinary
    prose replies (even ones that happen to contain a brace) are left untouched.
    """
    if not text:
        return None
    t = _strip(text)
    if not (t.startswith("{") and t.endswith("}")):
        return None
    try:
        data = json.loads(t)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("action")
    if not isinstance(raw, list) or not raw:
        return None
    actions: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or len(item) != 1:
            return None
        name, args = next(iter(item.items()))
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        actions.append({name: args})
    return actions
