"""Base protocol for LLM providers."""

from typing import Any, Callable, Optional, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from .messages import BaseMessage, ToolTurn

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class BaseChatModel(Protocol):
    """Protocol that all LLM providers must implement."""

    model: str

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[T] | None = None,
    ) -> str | T:
        """Invoke the LLM.

        Args:
            messages: List of conversation messages
            output_format: Optional Pydantic model for structured output

        Returns:
            String response or parsed Pydantic model
        """
        ...

    def with_structured_output(
        self, output_model: type[T], include_raw: bool = False
    ) -> Any:
        """Return a wrapper for structured output.

        Args:
            output_model: Pydantic model for output
            include_raw: If True, return {"parsed": result, "raw": ...}

        Returns:
            Wrapper object with ainvoke method
        """
        ...

    async def ainvoke_tools(
        self,
        messages: list[BaseMessage],
        tools: list[dict],
        on_event: Optional[Callable[[Any], None]] = None,
    ) -> ToolTurn:
        """Invoke the LLM with native provider tool-calling.

        Args:
            messages: Conversation messages (may include ToolResultMessage).
            tools: Provider-agnostic tool schemas, e.g. from
                ``ToolRegistry.tool_schemas()`` — ``[{name, description,
                input_schema}]``.
            on_event: Optional callback invoked with normalized ``PoriEvent``s
                (text_delta, tool_call_start, ...) as they stream. When provided,
                the provider streams; when ``None`` the call is a single
                non-streaming request. Providers without streaming may emit only
                a final text_delta.

        Returns:
            A ToolTurn with the assistant's text and any requested tool calls.
        """
        ...
