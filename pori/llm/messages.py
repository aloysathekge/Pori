"""Simple message types for LLM conversations."""

from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field


class SystemMessage(BaseModel):
    """System message."""

    role: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    """User message."""

    role: Literal["user"] = "user"
    content: str


class ToolCall(BaseModel):
    """A single native tool call requested by the model."""

    id: str = ""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class AssistantMessage(BaseModel):
    """Assistant message.

    ``tool_calls`` is populated only in native tool-calling mode; in the legacy
    JSON-envelope mode the assistant reply is plain ``content``.
    """

    role: Literal["assistant"] = "assistant"
    content: str = ""
    tool_calls: List[ToolCall] = Field(default_factory=list)


class ToolResultMessage(BaseModel):
    """The result of executing a tool, fed back to the model (native mode)."""

    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str


class ToolTurn(BaseModel):
    """One native tool-calling turn: assistant text plus requested tool calls."""

    text: str = ""
    tool_calls: List[ToolCall] = Field(default_factory=list)


class Usage(BaseModel):
    """Normalized token usage for one LLM call (provider-agnostic).

    Provider wrappers stash raw, provider-shaped usage dicts; ``normalize_usage``
    maps them here so the agent loop never branches on provider-specific keys.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


def normalize_usage(raw: "Dict[str, Any] | None") -> Usage:
    """Map a provider-shaped usage dict to a normalized :class:`Usage`.

    Centralizes provider-key knowledge (Anthropic ``input``/``output_tokens`` +
    cache keys; OpenAI/Google ``prompt``/``completion``/``total_tokens``) in the
    llm layer, so the agent loop reads normalized fields instead of branching on
    provider keys (AC-4).
    """
    if not isinstance(raw, dict):
        return Usage()

    def _i(*keys: str) -> int:
        for k in keys:
            value = raw.get(k)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return 0
        return 0

    input_tokens = _i("input_tokens", "prompt_tokens")
    output_tokens = _i("output_tokens", "completion_tokens")
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=_i("total_tokens") or (input_tokens + output_tokens),
        cache_read_tokens=_i("cache_read_input_tokens", "cache_read_tokens"),
        cache_write_tokens=_i("cache_creation_input_tokens", "cache_write_tokens"),
    )


# Union type for type hints
BaseMessage = Union[
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
]
