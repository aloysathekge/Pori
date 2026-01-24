"""Base protocol for LLM providers."""

from typing import Any, Protocol, TypeVar, runtime_checkable
from pydantic import BaseModel

from .messages import BaseMessage

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
