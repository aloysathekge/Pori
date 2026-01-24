"""Anthropic chat model."""

from typing import Any, Generic, TypeVar
from pydantic import BaseModel
from anthropic import AsyncAnthropic

from .messages import BaseMessage, SystemMessage

T = TypeVar("T", bound=BaseModel)


class ChatAnthropic:
    """Anthropic chat model wrapper."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[T] | None = None,
    ) -> str | T:
        """Invoke Anthropic model."""
        # Extract system message and convert others
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
            else:
                anthropic_messages.append({"role": msg.role, "content": msg.content})

        # Build request
        request: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens,
        }
        if system_prompt:
            request["system"] = system_prompt
        if self.temperature > 0:
            request["temperature"] = self.temperature

        if output_format is None:
            # Text response
            response = await self._client.messages.create(**request)
            return response.content[0].text
        else:
            # Structured output via tool use
            schema = output_format.model_json_schema()
            if "title" in schema:
                del schema["title"]

            request["tools"] = [
                {
                    "name": output_format.__name__,
                    "description": f"Extract {output_format.__name__}",
                    "input_schema": schema,
                }
            ]
            request["tool_choice"] = {"type": "tool", "name": output_format.__name__}

            response = await self._client.messages.create(**request)

            for block in response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    return output_format.model_validate(block.input)

            raise ValueError("No structured output in response")

    def with_structured_output(
        self, output_model: type[T], include_raw: bool = False
    ) -> "StructuredWrapper[T]":
        """Return a wrapper that always uses structured output."""
        return StructuredWrapper(self, output_model, include_raw)


class StructuredWrapper(Generic[T]):
    """Wrapper for structured output calls."""

    def __init__(
        self,
        llm: ChatAnthropic,
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
