"""Context-window policy separated from durable memory storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field


class ContextDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_messages: int
    included_messages: int
    dropped_messages: int
    estimated_tokens: int
    token_budget: int
    summary_included: bool = False
    recent_tail_preserved: int = 0
    reason: str = "within_budget"


class ContextWindow(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: Tuple[Dict[str, Any], ...]
    diagnostics: ContextDiagnostics


class ContextEngine(ABC):
    """Select prompt context without owning durable memory lifecycle."""

    @abstractmethod
    def build(
        self,
        memory: Any,
        *,
        max_tokens: int,
        reserve_tokens: int,
    ) -> ContextWindow:
        raise NotImplementedError


class DefaultContextEngine(ContextEngine):
    """Behavior-equivalent adapter around Pori's existing windowing policy."""

    def build(
        self,
        memory: Any,
        *,
        max_tokens: int,
        reserve_tokens: int,
    ) -> ContextWindow:
        budget = max(200, int(max_tokens) - int(reserve_tokens))
        eligible = [
            message
            for message in getattr(memory, "messages", [])
            if getattr(message, "role", None) in {"user", "assistant"}
        ]
        messages: List[Dict[str, Any]] = memory.get_token_limited_messages(
            max_tokens=max_tokens,
            reserve_tokens=reserve_tokens,
            include_summary_message=True,
        )
        included_regular = [item for item in messages if item.get("role") != "system"]
        summary_included = any(item.get("role") == "system" for item in messages)
        estimated = sum(
            memory.estimate_tokens(str(item.get("content", ""))) for item in messages
        )
        dropped = max(0, len(eligible) - len(included_regular))
        return ContextWindow(
            messages=tuple(messages),
            diagnostics=ContextDiagnostics(
                input_messages=len(eligible),
                included_messages=len(included_regular),
                dropped_messages=dropped,
                estimated_tokens=estimated,
                token_budget=budget,
                summary_included=summary_included,
                recent_tail_preserved=len(included_regular),
                reason="compacted" if dropped else "within_budget",
            ),
        )


__all__ = [
    "ContextDiagnostics",
    "ContextEngine",
    "ContextWindow",
    "DefaultContextEngine",
]
