"""Google Gemini chat model."""

import json
from typing import Any, Generic, TypeVar

from google import genai
from pydantic import BaseModel

from .messages import BaseMessage

T = TypeVar("T", bound=BaseModel)


class ChatGoogle:
    """Google Gemini chat model wrapper."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = genai.Client(api_key=api_key)
        self.last_usage: dict[str, Any] | None = None

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[T] | None = None,
    ) -> str | T:
        """Invoke Gemini model."""
        # Separate system instruction from conversation messages
        system_instruction = None
        contents = []
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                role = "model" if msg.role == "assistant" else "user"
                contents.append(
                    genai.types.Content(
                        role=role,
                        parts=[genai.types.Part(text=msg.content)],
                    )
                )

        config = genai.types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            system_instruction=system_instruction,
        )

        if output_format is not None:
            config.response_mime_type = "application/json"
            config.response_schema = output_format

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        # Capture usage
        try:
            usage = response.usage_metadata
            if usage:
                self.last_usage = {
                    "prompt_tokens": usage.prompt_token_count or 0,
                    "completion_tokens": usage.candidates_token_count or 0,
                    "total_tokens": usage.total_token_count or 0,
                }
        except Exception:
            self.last_usage = None

        text = response.text or ""

        if output_format is not None:
            return output_format.model_validate_json(text)

        return text

    def with_structured_output(
        self, output_model: type[T], include_raw: bool = False
    ) -> "StructuredWrapper[T]":
        """Return a wrapper that always uses structured output."""
        return StructuredWrapper(self, output_model, include_raw)


class StructuredWrapper(Generic[T]):
    """Wrapper for structured output calls."""

    def __init__(
        self,
        llm: ChatGoogle,
        output_model: type[T],
        include_raw: bool = False,
    ):
        self._llm = llm
        self._output_model = output_model
        self._include_raw = include_raw

    async def ainvoke(self, messages: list[BaseMessage]) -> dict[str, Any] | T:
        """Invoke with structured output."""
        result = await self._llm.ainvoke(messages, output_format=self._output_model)
        if self._include_raw:
            return {"parsed": result, "raw": None}
        return result
