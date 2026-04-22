"""OpenRouter chat model.

OpenRouter (https://openrouter.ai) exposes an OpenAI-compatible API that
routes to many hosted models — including open-source models (Llama, Mistral,
Qwen, DeepSeek, etc.). Because the wire format matches OpenAI's, we subclass
``ChatOpenAI`` and only override the client construction to point at
OpenRouter's base URL and carry the optional attribution headers.
"""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI

from .openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ChatOpenRouter(ChatOpenAI):
    """OpenRouter chat model wrapper (OpenAI-compatible)."""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str | None = None,
        http_referer: str | None = None,
        x_title: str | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_usage: dict[str, Any] | None = None

        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY")
        resolved_base = base_url or OPENROUTER_BASE_URL

        # OpenRouter encourages (but does not require) attribution headers
        # for rankings on openrouter.ai/rankings.
        default_headers: dict[str, str] = {}
        referer = http_referer or os.getenv("OPENROUTER_HTTP_REFERER")
        title = x_title or os.getenv("OPENROUTER_X_TITLE")
        if referer:
            default_headers["HTTP-Referer"] = referer
        if title:
            default_headers["X-Title"] = title

        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=resolved_base,
            default_headers=default_headers or None,
        )
