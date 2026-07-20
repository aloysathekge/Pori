"""Portable runtime identity, evidence, and fingerprint contracts."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def fail_open(
    what: str, logger: logging.Logger, *, level: int = logging.DEBUG
) -> Iterator[None]:
    """Swallow exceptions from non-essential bookkeeping (ADR 0004): the
    run must survive metrics/checkpoint/journal failures. Logs and moves on."""
    try:
        yield
    except Exception:
        logger.log(level, "fail-open: %s failed", what, exc_info=True)


def stable_fingerprint(value: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for JSON-compatible data."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ExecutionBudget(BaseModel):
    """Limits shared by a run and, later, its child runs."""

    model_config = ConfigDict(frozen=True)

    max_steps: Optional[int] = Field(default=None, ge=1)
    max_tool_calls: Optional[int] = Field(default=None, ge=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_cost_usd: Optional[float] = Field(default=None, ge=0.0)
    max_duration_seconds: Optional[float] = Field(default=None, gt=0.0)


BUDGET_EXHAUSTION_CODES = frozenset(
    {
        "max_steps",
        "max_tool_calls",
        "max_tokens",
        "max_cost_usd",
        "max_duration_seconds",
        "unpriced_model",
    }
)


class BudgetExceeded(RuntimeError):
    """A named execution ceiling was reached."""

    def __init__(self, message: str, *, code: str = "budget_exhausted"):
        self.code = code
        super().__init__(message)


class BudgetLedger:
    """Mutable shared consumption ledger for a parent run and its children."""

    def __init__(
        self,
        budget: ExecutionBudget,
        *,
        initial_usage: Optional[Mapping[str, Any]] = None,
    ):
        self.budget = budget
        usage = initial_usage or {}
        self.steps_used = max(0, int(usage.get("steps_used") or 0))
        self.tool_calls_used = max(0, int(usage.get("tool_calls_used") or 0))
        self.llm_calls_used = max(0, int(usage.get("llm_calls_used") or 0))
        self.input_tokens_used = max(0, int(usage.get("input_tokens_used") or 0))
        self.output_tokens_used = max(0, int(usage.get("output_tokens_used") or 0))
        self.cache_read_tokens_used = max(
            0, int(usage.get("cache_read_tokens_used") or 0)
        )
        self.cache_write_tokens_used = max(
            0, int(usage.get("cache_write_tokens_used") or 0)
        )
        self.tokens_used = max(0, int(usage.get("tokens_used") or 0))
        self.cost_used_usd = max(0.0, float(usage.get("cost_used_usd") or 0.0))
        self.unpriced_llm_calls = max(0, int(usage.get("unpriced_llm_calls") or 0))
        self._duration_used_before_start = max(
            0.0, float(usage.get("duration_seconds_used") or 0.0)
        )
        # Wall-clock deadline (max_duration_seconds): armed by the first
        # start_clock() so parent + children share one clock.
        self._clock_started_at: Optional[float] = None
        self._lock = threading.Lock()

    def start_clock(self) -> None:
        """Arm the wall-clock deadline. Idempotent — first caller wins, so a
        ledger shared with child runs measures from the parent's start."""
        with self._lock:
            if self._clock_started_at is None:
                self._clock_started_at = time.monotonic()

    def _check_deadline_locked(self) -> None:
        if (
            self.budget.max_duration_seconds is not None
            and self._clock_started_at is not None
            and self._duration_used_before_start
            + (time.monotonic() - self._clock_started_at)
            > self.budget.max_duration_seconds
        ):
            raise BudgetExceeded(
                "Duration budget exceeded", code="max_duration_seconds"
            )

    def check_deadline(self) -> None:
        """Fail before another model or tool action after the wall deadline."""
        with self._lock:
            self._check_deadline_locked()

    def check_model_call_allowed(self) -> None:
        """Fail before a provider call when no measurable budget remains."""
        with self._lock:
            self._check_deadline_locked()
            if (
                self.budget.max_tokens is not None
                and self.tokens_used >= self.budget.max_tokens
            ):
                raise BudgetExceeded("Token budget exhausted", code="max_tokens")
            if (
                self.budget.max_cost_usd is not None
                and self.cost_used_usd >= self.budget.max_cost_usd
            ):
                raise BudgetExceeded("Cost budget exhausted", code="max_cost_usd")
            if self.budget.max_cost_usd is not None and self.unpriced_llm_calls:
                raise BudgetExceeded(
                    "Cost budget cannot be verified for this model",
                    code="unpriced_model",
                )

    def consume_step(self, count: int = 1) -> None:
        with self._lock:
            self._check_deadline_locked()
            next_value = self.steps_used + count
            if self.budget.max_steps is not None and next_value > self.budget.max_steps:
                raise BudgetExceeded("Step budget exceeded", code="max_steps")
            self.steps_used = next_value

    def consume_tool_call(self, count: int = 1) -> None:
        """Reserve tool-call capacity before dispatching a model action."""
        if count <= 0:
            return
        with self._lock:
            self._check_deadline_locked()
            next_value = self.tool_calls_used + count
            if (
                self.budget.max_tool_calls is not None
                and next_value > self.budget.max_tool_calls
            ):
                raise BudgetExceeded("Tool-call budget exceeded", code="max_tool_calls")
            self.tool_calls_used = next_value

    def record_llm_call(self) -> None:
        """Record a completed provider call, including calls with no usage."""
        with self._lock:
            self.llm_calls_used += 1
            self._check_deadline_locked()

    def consume_tokens(
        self,
        count: int,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        if count <= 0:
            return
        with self._lock:
            self.input_tokens_used += max(0, input_tokens)
            self.output_tokens_used += max(0, output_tokens)
            self.cache_read_tokens_used += max(0, cache_read_tokens)
            self.cache_write_tokens_used += max(0, cache_write_tokens)
            next_value = self.tokens_used + count
            # The provider call already happened. Preserve actual consumption
            # even when it crossed the configured ceiling, then stop the loop.
            self.tokens_used = next_value
            self._check_deadline_locked()
            if (
                self.budget.max_tokens is not None
                and next_value > self.budget.max_tokens
            ):
                raise BudgetExceeded("Token budget exceeded", code="max_tokens")

    def consume_cost(self, amount_usd: float) -> None:
        if amount_usd <= 0:
            return
        with self._lock:
            next_value = self.cost_used_usd + amount_usd
            # Cost is known only after provider usage arrives. Record the real
            # charge before stopping so receipts never under-report an overage.
            self.cost_used_usd = next_value
            self._check_deadline_locked()
            if (
                self.budget.max_cost_usd is not None
                and next_value > self.budget.max_cost_usd
            ):
                raise BudgetExceeded("Cost budget exceeded", code="max_cost_usd")

    def record_unpriced_llm_call(self) -> None:
        """Fail closed when a cost ceiling cannot be verified for the model."""
        with self._lock:
            self.unpriced_llm_calls += 1
            if self.budget.max_cost_usd is not None:
                raise BudgetExceeded(
                    "Cost budget cannot be verified for this model",
                    code="unpriced_model",
                )

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "max_steps": self.budget.max_steps,
                "max_tool_calls": self.budget.max_tool_calls,
                "max_tokens": self.budget.max_tokens,
                "max_cost_usd": self.budget.max_cost_usd,
                "max_duration_seconds": self.budget.max_duration_seconds,
                "steps_used": self.steps_used,
                "tool_calls_used": self.tool_calls_used,
                "llm_calls_used": self.llm_calls_used,
                "input_tokens_used": self.input_tokens_used,
                "output_tokens_used": self.output_tokens_used,
                "cache_read_tokens_used": self.cache_read_tokens_used,
                "cache_write_tokens_used": self.cache_write_tokens_used,
                "tokens_used": self.tokens_used,
                "cost_used_usd": self.cost_used_usd,
                "unpriced_llm_calls": self.unpriced_llm_calls,
                "duration_seconds_used": (
                    self._duration_used_before_start
                    + (time.monotonic() - self._clock_started_at)
                    if self._clock_started_at is not None
                    else self._duration_used_before_start
                ),
            }


class RunCancelled(Exception):
    """The run's CancellationToken fired mid-step: the in-flight LLM call was
    aborted / remaining actions skipped. The agent loop catches this to wind
    down cleanly (state.stopped) instead of recording a step failure."""


class CancellationToken:
    """Shared cooperative cancellation signal."""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


class RunContext(BaseModel):
    """Immutable identity and authority context for one agent run."""

    model_config = ConfigDict(frozen=True)

    organization_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    event_id: Optional[str] = None
    workspace_id: Optional[str] = None
    permissions: Tuple[str, ...] = ()
    credential_scope: Optional[str] = None
    isolation_profile: str = "local"
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    metadata: Tuple[Tuple[str, str], ...] = ()

    @classmethod
    def local(
        cls,
        *,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: str = "local-user",
    ) -> "RunContext":
        generated_run_id = run_id or f"run_{uuid.uuid4().hex[:16]}"
        return cls(
            organization_id=f"local:{user_id}",
            user_id=user_id,
            agent_id=agent_id or generated_run_id,
            session_id=session_id or generated_run_id,
            run_id=generated_run_id,
        )


class ReceiptStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    STAGED = "staged"
    REUSED = "reused"


class ToolExecutionReceipt(BaseModel):
    """Auditable evidence for an attempted or reused tool action."""

    receipt_id: str = Field(default_factory=lambda: f"rcpt_{uuid.uuid4().hex[:16]}")
    run_id: str
    tool_name: str
    status: ReceiptStatus
    backend: str = "pori"
    parameters_fingerprint: str
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime = Field(default_factory=utc_now)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    error: Optional[str] = None
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChildRunRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    parent_run_id: str
    child_run_id: str = Field(default_factory=lambda: f"child_{uuid.uuid4().hex[:16]}")
    task: str
    agent_id: str
    allowed_tools: Tuple[str, ...] = ()
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    idempotency_key: Optional[str] = None


class ChildRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    parent_run_id: str
    child_run_id: str
    completed: bool
    final_answer: Optional[str] = None
    reasoning: Optional[str] = None
    steps_taken: int = 0
    usage: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


__all__ = [
    "BUDGET_EXHAUSTION_CODES",
    "ExecutionBudget",
    "BudgetExceeded",
    "BudgetLedger",
    "CancellationToken",
    "ChildRunRequest",
    "ChildRunResult",
    "ReceiptStatus",
    "RunContext",
    "ToolExecutionReceipt",
    "fail_open",
    "stable_fingerprint",
    "utc_now",
]
