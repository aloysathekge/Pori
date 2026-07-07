"""Cross-provider failover chain (Hermes gap Tier 1.3).

Wraps an ordered list of chat models. When the active model fails with a
provider-unavailability error — auth, billing, rate limit, overload, 5xx,
timeout (i.e. after that model's own retry/backoff is exhausted) — the chain
advances to the next model and the same call runs there. The switch is
STICKY: once a model is passed over, later calls start from the survivor, so
a dead primary is not re-tried on every step.

Deliberately NOT failover triggers: CONTEXT_OVERFLOW (follows the prompt, not
the provider — the agent's compression recovery owns it) and
CONTENT_POLICY_BLOCKED / UNKNOWN (may reproduce anywhere; propagating is more
honest than silently burning the whole chain).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Sequence

from .base import BaseChatModel
from .error_classifier import FailoverReason, classify_error
from .messages import BaseMessage, ToolTurn

logger = logging.getLogger(__name__)

FAILOVER_REASONS = frozenset(
    {
        FailoverReason.AUTH,
        FailoverReason.BILLING,
        FailoverReason.RATE_LIMIT,
        FailoverReason.OVERLOADED,
        FailoverReason.SERVER_ERROR,
        FailoverReason.TIMEOUT,
    }
)


class FailoverChatModel:
    """A chat model backed by an ordered chain of real models.

    Implements the same surface as a provider wrapper (``ainvoke``,
    ``ainvoke_tools``, ``model``, ``last_usage``, ...) by proxying to the
    currently active model, so the agent loop and metrics never know a
    failover happened beyond the model id changing.
    """

    def __init__(
        self,
        models: Sequence[BaseChatModel],
        labels: Optional[Sequence[str]] = None,
    ):
        if not models:
            raise ValueError("FailoverChatModel needs at least one model")
        self._models: List[BaseChatModel] = list(models)
        self._labels: List[str] = list(labels or [])
        while len(self._labels) < len(self._models):
            model = self._models[len(self._labels)]
            self._labels.append(
                f"{type(model).__name__}:{getattr(model, 'model', '?')}"
            )
        self._active = 0

    @property
    def active_model(self) -> BaseChatModel:
        return self._models[self._active]

    @property
    def active_label(self) -> str:
        return self._labels[self._active]

    def __getattr__(self, name: str) -> Any:
        # Everything not defined here (model, max_tokens, last_usage,
        # reasoning_mode, with_structured_output, ...) proxies to the active
        # model, so downstream getattr-style consumers keep working.
        return getattr(self._models[self._active], name)

    async def _call_with_failover(self, invoke: Callable[[BaseChatModel], Any]) -> Any:
        last_exc: Optional[BaseException] = None
        for index in range(self._active, len(self._models)):
            model = self._models[index]
            try:
                result = await invoke(model)
                self._active = index
                return result
            except BaseException as exc:
                classified = classify_error(exc)
                is_last = index == len(self._models) - 1
                if classified.reason not in FAILOVER_REASONS or is_last:
                    self._active = index
                    raise
                logger.warning(
                    "LLM failover: %s failed (%s) — switching to %s",
                    self._labels[index],
                    classified.reason.value,
                    self._labels[index + 1],
                )
                last_exc = exc
        # Unreachable (the last model either returned or re-raised), but keep
        # a defensive raise so a logic slip can never return None silently.
        raise last_exc if last_exc else RuntimeError("empty failover chain")

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: Any = None,
    ) -> Any:
        if output_format is None:
            return await self._call_with_failover(lambda m: m.ainvoke(messages))
        return await self._call_with_failover(
            lambda m: m.ainvoke(messages, output_format=output_format)
        )

    async def ainvoke_tools(
        self,
        messages: list[BaseMessage],
        tools: list[dict],
        on_event: Optional[Callable[[Any], None]] = None,
    ) -> ToolTurn:
        return await self._call_with_failover(
            lambda m: (
                m.ainvoke_tools(messages, tools, on_event=on_event)
                if on_event is not None
                else m.ainvoke_tools(messages, tools)
            )
        )


__all__ = ["FAILOVER_REASONS", "FailoverChatModel"]
