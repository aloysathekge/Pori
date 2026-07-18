"""OpenAI chat model."""

import json
from typing import Any, Callable, Generic, Optional, TypeVar, cast

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from ..observability.events import TEXT_DELTA, THINKING_DELTA, PoriEvent
from .messages import (
    AssistantMessage,
    BaseMessage,
    DocumentBlock,
    ImageBlock,
    MessageContent,
    TextBlock,
    ToolCall,
    ToolResultMessage,
    ToolTurn,
)
from .reasoning import StreamingThinkScrubber
from .retry import RetryConfig, retry_async
from .structured_output import StructuredOutputPolicy


def _to_openai_content(content: MessageContent) -> Any:
    """Map message content to OpenAI's shape (str passes through)."""
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextBlock):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            url = (
                block.url
                if block.source == "url"
                else f"data:{block.media_type};base64,{block.data}"
            )
            parts.append({"type": "image_url", "image_url": {"url": url}})
        elif isinstance(block, DocumentBlock):
            parts.append(
                {
                    "type": "file",
                    "file": {
                        "filename": block.name or "document.pdf",
                        "file_data": f"data:{block.media_type};base64,{block.data}",
                    },
                }
            )
    return parts or ""


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
        reasoning_mode: str = "none",
        **kwargs: Any,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_mode = reasoning_mode
        self._client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()
        # Last usage metadata from the most recent call (for metrics)
        self.last_usage: dict[str, Any] | None = None
        # Retry transient API failures (rate limits, timeouts, 5xx) with backoff.
        # Subclasses (OpenRouter, Fireworks) that skip super().__init__ fall back
        # to env defaults via getattr in ainvoke.
        self._retry_config = RetryConfig.from_env()

    def _structured_output_policy(self) -> StructuredOutputPolicy:
        """Return the structured-output contract for this provider/model."""
        return StructuredOutputPolicy()

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[T] | None = None,
    ) -> str | T:
        """Invoke OpenAI model."""
        openai_messages = [
            {"role": m.role, "content": _to_openai_content(m.content)} for m in messages
        ]
        output_schema: dict[str, Any] | None = None
        structured_policy: StructuredOutputPolicy | None = None
        if output_format is not None:
            structured_policy = self._structured_output_policy()
            output_schema = structured_policy.adapt_schema(
                output_format.model_json_schema()
            )
            openai_messages = structured_policy.prepare_messages(
                openai_messages,
                output_schema,
            )

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
            assert output_schema is not None
            assert structured_policy is not None
            request["response_format"] = structured_policy.response_format(
                output_schema
            )
            request.update(structured_policy.request_options)
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
        self,
        messages: list[BaseMessage],
        tools: list[dict],
        on_event: Optional[Callable[[PoriEvent], None]] = None,
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
                openai_messages.append(
                    {"role": m.role, "content": _to_openai_content(m.content)}
                )

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

        # Streaming path: emit normalized events (text deltas + instant
        # tool_call_start) while assembling the full ToolTurn from the stream.
        if on_event is not None:
            return await self._stream_tools(request, on_event)

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

    async def _stream_tools(
        self, request: dict[str, Any], on_event: Callable[[PoriEvent], None]
    ) -> ToolTurn:
        """Stream a tool-call turn, emitting normalized PoriEvents.

        Per the accumulator rule: text streams live (``text_delta``); tool-call
        arguments are buffered silently (partial JSON isn't parseable), but a
        ``tool_call_start`` fires the instant the tool name is known — before its
        args finish — so tool calls feel instant.
        """
        request = {
            **request,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        stream = await retry_async(
            lambda: self._client.chat.completions.create(**request),
            getattr(self, "_retry_config", None),
            label="openai",
        )

        def _emit(event: PoriEvent) -> None:
            try:
                on_event(event)
            except Exception:
                # A consumer error must never break the run.
                pass

        text_parts: list[str] = []
        # index -> {id, name, args} accumulated across delta chunks.
        acc: dict[int, dict[str, str]] = {}
        usage = None

        # Reasoning tier: 'native' = a separate reasoning channel; 'tagged' =
        # inline <think>...</think> in the text; 'none' = plain text only.
        mode = getattr(self, "reasoning_mode", "none")
        scrubber = StreamingThinkScrubber() if mode == "tagged" else None

        def _emit_segment(kind: str, seg: str) -> None:
            if not seg:
                return
            if kind == "thinking":
                _emit(PoriEvent(THINKING_DELTA, {"text": seg}))
            else:  # visible answer text
                text_parts.append(seg)
                _emit(PoriEvent(TEXT_DELTA, {"text": seg}))

        async for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = choices[0].delta
            if mode == "native":
                reasoning = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )
                if reasoning:
                    _emit(PoriEvent(THINKING_DELTA, {"text": reasoning}))
            content = getattr(delta, "content", None)
            if content:
                if scrubber is not None:
                    for kind, seg in scrubber.feed(content):
                        _emit_segment(kind, seg)
                else:
                    _emit_segment("text", content)
            for tc in getattr(delta, "tool_calls", None) or []:
                idx = getattr(tc, "index", 0) or 0
                slot = acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"] += fn.arguments

        if scrubber is not None:
            for kind, seg in scrubber.flush():
                _emit_segment(kind, seg)

        if usage is not None:
            self.last_usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        else:
            self.last_usage = None

        tool_calls: list[ToolCall] = []
        for idx in sorted(acc):
            slot = acc[idx]
            if not slot["name"]:
                continue
            try:
                args = json.loads(slot["args"]) if slot["args"] else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append(
                ToolCall(
                    id=slot["id"],
                    name=slot["name"],
                    arguments=args if isinstance(args, dict) else {},
                )
            )
        return ToolTurn(text="".join(text_parts), tool_calls=tool_calls)

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
