"""OpenAI chat model."""

from typing import Any, Generic, TypeVar
from pydantic import BaseModel
from openai import AsyncOpenAI

from .messages import BaseMessage

T = TypeVar("T", bound=BaseModel)


class ChatOpenAI:
    """OpenAI chat model wrapper."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[T] | None = None,
    ) -> str | T:
        """Invoke OpenAI model."""
        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        request: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if output_format is None:
            response = await self._client.chat.completions.create(**request)
            return response.choices[0].message.content or ""
        else:
            # Use response_format for structured output
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "strict": True,
                    "schema": output_format.model_json_schema(),
                },
            }
            response = await self._client.chat.completions.create(**request)
            content = response.choices[0].message.content
            return output_format.model_validate_json(content)

    def with_structured_output(
        self, output_model: type[T], include_raw: bool = False
    ) -> "StructuredWrapper[T]":
        """Return a wrapper that always uses structured output."""
        return StructuredWrapper(self, output_model, include_raw)


class StructuredWrapper(Generic[T]):
    """Wrapper for structured output calls."""

    def __init__(
        self,
        llm: ChatOpenAI,
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
