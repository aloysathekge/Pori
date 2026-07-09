"""Google Gemini chat model."""

import json
from typing import Any, Callable, Generic, Optional, TypeVar, cast

from google import genai
from pydantic import BaseModel

from .messages import (
    BaseMessage,
    DocumentBlock,
    ImageBlock,
    MessageContent,
    TextBlock,
    ToolCall,
    ToolResultMessage,
    ToolTurn,
)
from .retry import RetryConfig, retry_async


def _to_google_parts(content: MessageContent) -> "list[genai.types.Part]":
    """Map message content to Gemini Parts (str becomes one text part).

    URL image sources degrade to a text placeholder — Gemini inline content
    requires bytes, and silently dropping the image would hide information.
    """
    if isinstance(content, str):
        return [genai.types.Part(text=content)]
    import base64 as _base64

    parts: "list[genai.types.Part]" = []
    for block in content:
        if isinstance(block, TextBlock):
            parts.append(genai.types.Part(text=block.text))
        elif isinstance(block, ImageBlock):
            if block.source == "base64" and block.data:
                parts.append(
                    genai.types.Part.from_bytes(
                        data=_base64.b64decode(block.data),
                        mime_type=block.media_type,
                    )
                )
            else:
                parts.append(genai.types.Part(text=f"[image at {block.url}]"))
        elif isinstance(block, DocumentBlock):
            parts.append(
                genai.types.Part.from_bytes(
                    data=_base64.b64decode(block.data),
                    mime_type=block.media_type,
                )
            )
    return parts or [genai.types.Part(text="")]


T = TypeVar("T", bound=BaseModel)


def _sanitize_gemini_schema(obj: Any) -> Any:
    """Strip JSON-Schema keys the Gemini function API rejects (title, etc.)."""
    if isinstance(obj, dict):
        return {
            k: _sanitize_gemini_schema(v)
            for k, v in obj.items()
            if k not in ("title", "additionalProperties")
        }
    if isinstance(obj, list):
        return [_sanitize_gemini_schema(v) for v in obj]
    return obj


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
        # Retry transient API failures (rate limits, timeouts, 5xx) with backoff.
        self._retry_config = RetryConfig.from_env()

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
                        parts=_to_google_parts(msg.content),
                    )
                )

        config = genai.types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            system_instruction=system_instruction,
        )

        response = None
        if output_format is not None:
            config.response_mime_type = "application/json"
            config.response_schema = output_format
            try:
                response = await retry_async(
                    lambda: self._client.aio.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config,
                    ),
                    self._retry_config,
                    label="google",
                )
            except ValueError as e:
                # Gemini's schema transformer rejects JSON Schema features like
                # additionalProperties. Fall back to JSON-only mode and validate
                # locally with Pydantic.
                if (
                    "additionalProperties is not supported in the Gemini API"
                    not in str(e)
                ):
                    raise
                config.response_schema = None

        if response is None:
            response = await retry_async(
                lambda: self._client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                ),
                self._retry_config,
                label="google",
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
            if text.startswith("```"):
                text = text.strip().removeprefix("```json").removeprefix("```")
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            return output_format.model_validate_json(text)

        return text

    async def ainvoke_tools(
        self,
        messages: list[BaseMessage],
        tools: list[dict],
        on_event: Optional[Callable[[Any], None]] = None,
    ) -> ToolTurn:
        """Invoke Gemini with native function-calling.

        Native streaming isn't implemented here yet; when ``on_event`` is supplied
        the call runs non-streaming and the full text is delivered as one
        ``text_delta`` event.
        """
        system_instruction = None
        contents = []
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif isinstance(msg, ToolResultMessage):
                contents.append(
                    genai.types.Content(
                        role="user",
                        parts=[
                            genai.types.Part.from_function_response(
                                name=msg.tool_call_id,
                                response={"result": msg.content},
                            )
                        ],
                    )
                )
            else:
                role = "model" if msg.role == "assistant" else "user"
                contents.append(
                    genai.types.Content(
                        role=role,
                        parts=_to_google_parts(msg.content),
                    )
                )

        function_declarations = [
            genai.types.FunctionDeclaration.model_validate(
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": _sanitize_gemini_schema(t["input_schema"]),
                }
            )
            for t in tools
        ]
        config = genai.types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            system_instruction=system_instruction,
            tools=[genai.types.Tool(function_declarations=function_declarations)],
        )

        response = await retry_async(
            lambda: self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            ),
            self._retry_config,
            label="google",
        )

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

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for candidate in getattr(response, "candidates", None) or []:
            parts = getattr(getattr(candidate, "content", None), "parts", None) or []
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    tool_calls.append(
                        ToolCall(
                            id=getattr(fc, "id", "") or "",
                            name=getattr(fc, "name", "") or "",
                            arguments=dict(getattr(fc, "args", None) or {}),
                        )
                    )
                elif getattr(part, "text", None):
                    text_parts.append(part.text)
        turn = ToolTurn(
            text=" ".join(text_parts).strip(),
            tool_calls=tool_calls,
        )
        if on_event is not None and turn.text:
            from ..observability.events import TEXT_DELTA, PoriEvent
            from ..utils.action_decode import looks_like_action_envelope

            if not looks_like_action_envelope(turn.text):
                try:
                    on_event(PoriEvent(TEXT_DELTA, {"text": turn.text}))
                except Exception:
                    pass
        return turn

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
        return cast(T, result)
