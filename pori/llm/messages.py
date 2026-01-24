"""Simple message types for LLM conversations."""

from typing import Literal, Union
from pydantic import BaseModel


class SystemMessage(BaseModel):
    """System message."""

    role: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    """User message."""

    role: Literal["user"] = "user"
    content: str


class AssistantMessage(BaseModel):
    """Assistant message."""

    role: Literal["assistant"] = "assistant"
    content: str


# Union type for type hints
BaseMessage = Union[SystemMessage, UserMessage, AssistantMessage]
