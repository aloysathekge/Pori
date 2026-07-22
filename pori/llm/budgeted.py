"""One budget boundary around every model invocation in a Run."""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

from ..metrics import TokenUsage, estimate_llm_call_cost
from ..runtime import BudgetExceeded, BudgetLedger
from .base import BaseChatModel
from .messages import BaseMessage, ToolTurn, normalize_usage

T = TypeVar("T", bound=BaseModel)


class _BudgetedStructuredInvoker:
    def __init__(self, inner: Any, owner: "BudgetedChatModel"):
        self._inner = inner
        self._owner = owner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def ainvoke(self, messages: list[BaseMessage]) -> Any:
        self._owner.preflight()
        result = await self._inner.ainvoke(messages)
        self._owner.charge_completed_call()
        return result

    async def ainvoke_with_deltas(
        self,
        messages: list[BaseMessage],
        on_delta: Callable[[str], None],
    ) -> Any:
        self._owner.preflight()
        stream_invoke = getattr(self._inner, "ainvoke_with_deltas", None)
        if not callable(stream_invoke):
            result = await self._inner.ainvoke(messages)
        else:
            result = await stream_invoke(messages, on_delta)
        self._owner.charge_completed_call()
        return result


class BudgetedChatModel:
    """Transparent model proxy charging one shared :class:`BudgetLedger`.

    The wrapper sits below Agent planning, compression, validation, ordinary
    tool-calling, and Team coordination. This avoids separate call sites
    drifting into unmetered provider usage.
    """

    def __init__(self, inner: BaseChatModel, ledger: BudgetLedger):
        self._inner = inner
        self.budget_ledger = ledger
        self.model = str(getattr(inner, "model", ""))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def preflight(self) -> None:
        self.budget_ledger.check_model_call_allowed()

    def charge_completed_call(self) -> None:
        # A failover chain may switch its active model during the call.
        self.model = str(getattr(self._inner, "model", self.model))
        usage = normalize_usage(getattr(self._inner, "last_usage", None))
        tokens = TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        )
        cost = (
            estimate_llm_call_cost(self.model, tokens) if tokens.total_tokens else None
        )
        budget_error: BudgetExceeded | None = None

        try:
            self.budget_ledger.record_llm_call()
        except BudgetExceeded as exhausted:
            budget_error = exhausted
        try:
            self.budget_ledger.consume_tokens(
                tokens.total_tokens,
                input_tokens=tokens.input_tokens,
                output_tokens=tokens.output_tokens,
                cache_read_tokens=tokens.cache_read_tokens,
                cache_write_tokens=tokens.cache_write_tokens,
            )
        except BudgetExceeded as exhausted:
            if budget_error is None:
                budget_error = exhausted
        try:
            if cost is not None:
                self.budget_ledger.consume_cost(cost)
            else:
                # A completed call without known pricing (including missing
                # provider usage) cannot satisfy a configured cost ceiling.
                self.budget_ledger.record_unpriced_llm_call()
        except BudgetExceeded as exhausted:
            if budget_error is None:
                budget_error = exhausted
        if budget_error is not None:
            raise budget_error

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[T] | None = None,
    ) -> str | T:
        self.preflight()
        result: Any
        if output_format is None:
            result = await self._inner.ainvoke(messages)
        else:
            result = await self._inner.ainvoke(messages, output_format=output_format)
        self.charge_completed_call()
        return result

    async def ainvoke_tools(
        self,
        messages: list[BaseMessage],
        tools: list[dict],
        on_event: Optional[Callable[[Any], None]] = None,
    ) -> ToolTurn:
        self.preflight()
        if on_event is None:
            result = await self._inner.ainvoke_tools(messages, tools)
        else:
            result = await self._inner.ainvoke_tools(
                messages,
                tools,
                on_event=on_event,
            )
        self.charge_completed_call()
        return result

    def with_structured_output(
        self,
        output_model: type[T],
        include_raw: bool = False,
    ) -> Any:
        return _BudgetedStructuredInvoker(
            self._inner.with_structured_output(
                output_model,
                include_raw=include_raw,
            ),
            self,
        )


def ensure_budgeted_chat_model(
    model: BaseChatModel,
    ledger: BudgetLedger,
) -> BudgetedChatModel:
    if isinstance(model, BudgetedChatModel) and model.budget_ledger is ledger:
        return model
    return BudgetedChatModel(model, ledger)


__all__ = ["BudgetedChatModel", "ensure_budgeted_chat_model"]
