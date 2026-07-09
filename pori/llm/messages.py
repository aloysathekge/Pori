"""Simple message types for LLM conversations."""

import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    """One text segment of a multimodal message."""

    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    """One image segment of a multimodal message.

    ``source="base64"`` carries the payload inline in ``data``;
    ``source="url"`` points at a fetchable image. Provider adapters map this
    to their native shape (Anthropic image blocks, OpenAI ``image_url``,
    Google inline data). Not every provider supports URL sources — adapters
    degrade to a text placeholder rather than failing the call.
    """

    type: Literal["image"] = "image"
    source: Literal["base64", "url"] = "base64"
    media_type: str = "image/png"
    data: str = ""  # base64 payload when source="base64"
    url: str = ""  # location when source="url"

    @classmethod
    def from_bytes(cls, raw: bytes, media_type: str = "image/png") -> "ImageBlock":
        return cls(
            source="base64",
            media_type=media_type,
            data=base64.b64encode(raw).decode("ascii"),
        )

    @classmethod
    def from_file(cls, path: "str | Path") -> "ImageBlock":
        p = Path(path)
        media_type = mimetypes.guess_type(p.name)[0] or "image/png"
        return cls.from_bytes(p.read_bytes(), media_type=media_type)


class DocumentBlock(BaseModel):
    """One document segment (PDF) of a multimodal message.

    Providers accept PDFs natively: Anthropic ``document`` blocks, Gemini
    inline data, OpenAI ``file`` content parts (data URL). Non-PDF formats
    (docx, …) should be text-extracted by the caller instead."""

    type: Literal["document"] = "document"
    media_type: str = "application/pdf"
    data: str = ""  # base64 payload
    name: str = ""  # original filename, where the provider accepts one


ContentBlock = Union[TextBlock, ImageBlock, DocumentBlock]

# A message body: plain text (the overwhelmingly common case) or an ordered
# list of text/image blocks. Everything that only needs the words should go
# through content_text() instead of assuming str.
MessageContent = Union[str, List[ContentBlock]]


def content_text(content: MessageContent) -> str:
    """The textual portion of a message body (images contribute a marker)."""
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for block in content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ImageBlock):
            parts.append(f"[image: {block.media_type}]")
    return "\n".join(parts)


def content_has_images(content: MessageContent) -> bool:
    return not isinstance(content, str) and any(
        isinstance(block, ImageBlock) for block in content
    )


class SystemMessage(BaseModel):
    """System message."""

    role: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    """User message. ``content`` may be plain text or text+image blocks."""

    role: Literal["user"] = "user"
    content: MessageContent


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

__all__ = [
    "AssistantMessage",
    "BaseMessage",
    "ContentBlock",
    "ImageBlock",
    "MessageContent",
    "SystemMessage",
    "TextBlock",
    "ToolCall",
    "ToolResultMessage",
    "ToolTurn",
    "Usage",
    "UserMessage",
    "content_has_images",
    "content_text",
    "normalize_usage",
]
