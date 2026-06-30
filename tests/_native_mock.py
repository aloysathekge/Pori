"""Shared helper for native-mode test mocks.

Converts a legacy AgentOutput-shaped mock response ({current_state, action})
into a native ToolTurn, so existing test fixtures keep working under native
tool-calling without rewriting their action literals.
"""

from __future__ import annotations

from typing import Any

from pori.llm import ToolCall, ToolTurn


def tool_turn_from_response(response: Any) -> ToolTurn:
    """Build a ToolTurn from a mock response (parsed AgentOutput or content dict)."""
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        current_state = getattr(parsed, "current_state", {}) or {}
        action = getattr(parsed, "action", []) or []
    else:
        content = getattr(response, "content", response)
        if isinstance(content, dict):
            current_state = content.get("current_state", {}) or {}
            action = content.get("action", []) or []
        else:
            current_state, action = {}, []

    tool_calls = []
    for item in action:
        if isinstance(item, dict):
            for name, args in item.items():
                tool_calls.append(ToolCall(name=name, arguments=args or {}))
    text = current_state.get("next_goal", "") if isinstance(current_state, dict) else ""
    return ToolTurn(text=text, tool_calls=tool_calls)
