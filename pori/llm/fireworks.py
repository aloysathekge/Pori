"""Fireworks AI chat model.

Fireworks (https://fireworks.ai) hosts open-weight and partner models behind
an OpenAI-compatible API, so we subclass ``ChatOpenAI`` and only override
client construction to point at the Fireworks base URL.

Model slugs use the ``accounts/<account>/models/<model>`` form, e.g.
``accounts/fireworks/models/kimi-k2p6``.
"""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI

from .openai import ChatOpenAI

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


class ChatFireworks(ChatOpenAI):
    """Fireworks AI chat model wrapper (OpenAI-compatible)."""

    def __init__(
        self,
        model: str = "accounts/fireworks/models/kimi-k2p6",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_usage: dict[str, Any] | None = None

        resolved_key = api_key or os.getenv("FIREWORKS_API_KEY")
        resolved_base = (
            base_url or os.getenv("FIREWORKS_BASE_URL") or FIREWORKS_BASE_URL
        )

        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=resolved_base,
        )
