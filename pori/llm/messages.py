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


# Union type for type hints
BaseMessage = Union[
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
]
