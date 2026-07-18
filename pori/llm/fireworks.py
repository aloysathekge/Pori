"""Fireworks AI chat model.

Fireworks (https://fireworks.ai) hosts open-weight and partner models behind
an OpenAI-compatible API, so we subclass ``ChatOpenAI`` and only override
client construction to point at the Fireworks base URL.

Model slugs use the ``accounts/<account>/models/<model>`` form, e.g.
``accounts/fireworks/models/kimi-k2p6``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

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
        reasoning_mode: str = "none",
        **kwargs: Any,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_mode = reasoning_mode
        self.last_usage: dict[str, Any] | None = None

        resolved_key = api_key or os.getenv("FIREWORKS_API_KEY")
        resolved_base = (
            base_url or os.getenv("FIREWORKS_BASE_URL") or FIREWORKS_BASE_URL
        )

        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=resolved_base,
        )

    def _prepare_structured_messages(
        self,
        messages: list[dict[str, Any]],
        output_format: type[BaseModel],
    ) -> list[dict[str, Any]]:
        """Give Fireworks models the schema they enforce during generation.

        Fireworks recommends including the JSON schema in both the prompt and
        ``response_format``. Kimi otherwise may spend its content channel
        describing a schema it cannot see instead of returning the JSON body.
        """
        prepared = [dict(message) for message in messages]
        schema_instruction = (
            "\n\nReturn only JSON matching this required schema:\n"
            + json.dumps(
                output_format.model_json_schema(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        for message in reversed(prepared):
            if message.get("role") == "user" and isinstance(
                message.get("content"), str
            ):
                message["content"] += schema_instruction
                return prepared
        prepared.append({"role": "user", "content": schema_instruction.lstrip()})
        return prepared

    def _structured_request_options(self) -> dict[str, Any]:
        # Kimi K2.6 can otherwise place its analysis in the normal content
        # channel. Fireworks supports `none` for this family (including Turbo),
        # while some other hosted families reject it, so keep this targeted.
        if "kimi-k2p6" in self.model.lower():
            return {"reasoning_effort": "none"}
        return {}
