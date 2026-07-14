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
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_cost_usd: Optional[float] = Field(default=None, ge=0.0)
    max_duration_seconds: Optional[float] = Field(default=None, gt=0.0)


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    """Mutable shared consumption ledger for a parent run and its children."""

    def __init__(self, budget: ExecutionBudget):
        self.budget = budget
        self.steps_used = 0
        self.tokens_used = 0
        self.cost_used_usd = 0.0
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
            and time.monotonic() - self._clock_started_at
            > self.budget.max_duration_seconds
        ):
            raise BudgetExceeded("Duration budget exceeded")

    def consume_step(self, count: int = 1) -> None:
        with self._lock:
            self._check_deadline_locked()
            next_value = self.steps_used + count
            if self.budget.max_steps is not None and next_value > self.budget.max_steps:
                raise BudgetExceeded("Step budget exceeded")
            self.steps_used = next_value

    def consume_tokens(self, count: int) -> None:
        if count <= 0:
            return
        with self._lock:
            next_value = self.tokens_used + count
            if (
                self.budget.max_tokens is not None
                and next_value > self.budget.max_tokens
            ):
                raise BudgetExceeded("Token budget exceeded")
            self.tokens_used = next_value

    def consume_cost(self, amount_usd: float) -> None:
        if amount_usd <= 0:
            return
        with self._lock:
            next_value = self.cost_used_usd + amount_usd
            if (
                self.budget.max_cost_usd is not None
                and next_value > self.budget.max_cost_usd
            ):
                raise BudgetExceeded("Cost budget exceeded")
            self.cost_used_usd = next_value

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "max_steps": self.budget.max_steps,
                "max_tokens": self.budget.max_tokens,
                "max_cost_usd": self.budget.max_cost_usd,
                "max_duration_seconds": self.budget.max_duration_seconds,
                "steps_used": self.steps_used,
                "tokens_used": self.tokens_used,
                "cost_used_usd": self.cost_used_usd,
                "duration_seconds_used": (
                    time.monotonic() - self._clock_started_at
                    if self._clock_started_at is not None
                    else 0.0
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
