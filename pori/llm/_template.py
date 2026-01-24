"""Template for adding a new LLM provider.

Copy this file to create a new provider (e.g., google.py, azure.py).
Replace all instances of "ProviderName" with your provider name.
"""

from typing import Any, Generic, TypeVar
from pydantic import BaseModel
# TODO: Import your provider's SDK
# from provider_sdk import AsyncClient

from .messages import BaseMessage

T = TypeVar("T", bound=BaseModel)


class ChatProviderName:
    """ProviderName chat model wrapper."""

    def __init__(
        self,
        model: str = "default-model-name",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # TODO: Initialize your provider's SDK client
        # self._client = AsyncClient(api_key=api_key, model=model)

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[T] | None = None,
    ) -> str | T:
        """Invoke ProviderName model."""
        # Step 1: Convert universal messages to provider format
        provider_messages = []
        for msg in messages:
            # TODO: Convert to provider's message format
            # Example:
            # provider_messages.append({
            #     "role": msg.role,
            #     "content": msg.content
            # })
            pass

        # Step 2: Build request
        request: dict[str, Any] = {
            "model": self.model,
            # TODO: Add provider-specific request parameters
            # "messages": provider_messages,
            # "temperature": self.temperature,
            # "max_tokens": self.max_tokens,
        }

        # Step 3: Handle text vs structured output
        if output_format is None:
            # Text response path
            # TODO: Call provider API for text completion
            # response = await self._client.complete(**request)
            # return extract_text_from_response(response)
            raise NotImplementedError("Text completion not implemented")
        else:
            # Structured output path
            # TODO: Convert Pydantic model to provider's schema format
            schema = output_format.model_json_schema()
            
            # TODO: Add provider-specific structured output parameters
            # Example for OpenAI-style:
            # request["response_format"] = {
            #     "type": "json_schema",
            #     "json_schema": {"schema": schema}
            # }
            
            # TODO: Call provider API
            # response = await self._client.complete(**request)
            
            # TODO: Extract and parse structured result
            # content = extract_content_from_response(response)
            # return output_format.model_validate_json(content)
            raise NotImplementedError("Structured output not implemented")

    def with_structured_output(
        self, output_model: type[T], include_raw: bool = False
    ) -> "StructuredWrapper[T]":
        """Return a wrapper that always uses structured output."""
        return StructuredWrapper(self, output_model, include_raw)


class StructuredWrapper(Generic[T]):
    """Wrapper for structured output calls.
    
    This is the same for all providers - just copy as-is.
    """

    def __init__(
        self,
        llm: ChatProviderName,
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
