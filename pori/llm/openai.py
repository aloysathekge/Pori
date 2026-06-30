"""OpenAI chat model."""

import json
from typing import Any, Generic, TypeVar, cast

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from .messages import (
    AssistantMessage,
    BaseMessage,
    ToolCall,
    ToolResultMessage,
    ToolTurn,
)
from .retry import RetryConfig, retry_async

T = TypeVar("T", bound=BaseModel)


class StructuredOutputParseError(ValueError):
    """Raised when a structured response is present but invalid JSON/schema."""

    def __init__(self, message: str, *, raw_content: Any = None):
        super().__init__(message)
        self.raw_content = raw_content


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
        # Last usage metadata from the most recent call (for metrics)
        self.last_usage: dict[str, Any] | None = None
        # Retry transient API failures (rate limits, timeouts, 5xx) with backoff.
        # Subclasses (OpenRouter, Fireworks) that skip super().__init__ fall back
        # to env defaults via getattr in ainvoke.
        self._retry_config = RetryConfig.from_env()

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
            response = await retry_async(
                lambda: self._client.chat.completions.create(**request),
                getattr(self, "_retry_config", None),
                label="openai",
            )
            # Capture usage for metrics if available
            try:
                if getattr(response, "usage", None) is not None:
                    self.last_usage = {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(
                            response.usage, "completion_tokens", 0
                        ),
                        "total_tokens": getattr(response.usage, "total_tokens", 0),
                    }
            except Exception:
                self.last_usage = None
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
            response = await retry_async(
                lambda: self._client.chat.completions.create(**request),
                getattr(self, "_retry_config", None),
                label="openai",
            )

            # Capture usage for metrics if available
            try:
                if getattr(response, "usage", None) is not None:
                    self.last_usage = {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(
                            response.usage, "completion_tokens", 0
                        ),
                        "total_tokens": getattr(response.usage, "total_tokens", 0),
                    }
            except Exception:
                self.last_usage = None

            content = response.choices[0].message.content
            try:
                return output_format.model_validate_json(content)
            except ValidationError as exc:
                raise StructuredOutputParseError(
                    f"Structured output parse failed: {exc}",
                    raw_content=content,
                ) from exc

    async def ainvoke_tools(
        self, messages: list[BaseMessage], tools: list[dict]
    ) -> ToolTurn:
        """Invoke with native OpenAI-style tool-calling."""
        openai_messages: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m, ToolResultMessage):
                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.tool_call_id,
                        "content": m.content,
                    }
                )
            elif isinstance(m, AssistantMessage) and m.tool_calls:
                openai_messages.append(
                    {
                        "role": "assistant",
                        "content": m.content or None,
                        "tool_calls": [
                            {
                                "id": tc.id or f"call_{i}",
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for i, tc in enumerate(m.tool_calls)
                        ],
                    }
                )
            else:
                openai_messages.append({"role": m.role, "content": m.content})

        request: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ],
            "tool_choice": "auto",
        }
        response = await retry_async(
            lambda: self._client.chat.completions.create(**request),
            getattr(self, "_retry_config", None),
            label="openai",
        )
        try:
            if getattr(response, "usage", None) is not None:
                self.last_usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(
                        response.usage, "completion_tokens", 0
                    ),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }
        except Exception:
            self.last_usage = None

        message = response.choices[0].message
        tool_calls: list[ToolCall] = []
        for tc in getattr(message, "tool_calls", None) or []:
            raw_args = getattr(tc.function, "arguments", "") or "{}"
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append(
                ToolCall(
                    id=getattr(tc, "id", "") or "",
                    name=tc.function.name,
                    arguments=args if isinstance(args, dict) else {},
                )
            )
        return ToolTurn(text=message.content or "", tool_calls=tool_calls)

    def with_structured_output(
        self, output_model: type[T], include_raw: bool = False
    ) -> "StructuredWrapper[T]":
        """Return a wrapper that always uses structured output."""
        return StructuredWrapper(self, output_model, include_raw)


class StructuredWrapper(Generic[T]):
    """Wrapper for structured output calls."""

    def __init__(
        self,
        llm: Any,
        output_model: type[T],
        include_raw: bool = False,
    ):
        self._llm = llm
        self._output_model = output_model
        self._include_raw = include_raw

    async def ainvoke(self, messages: list[BaseMessage]) -> dict[str, Any] | T:
        """Invoke with structured output."""
        try:
            result = await self._llm.ainvoke(messages, output_format=self._output_model)
        except StructuredOutputParseError as exc:
            if self._include_raw:
                return {"parsed": None, "raw": exc.raw_content, "error": str(exc)}
            raise
        if self._include_raw:
            return {"parsed": result, "raw": None}
        return cast(T, result)
