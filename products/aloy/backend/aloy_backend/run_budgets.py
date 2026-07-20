"""Host-owned resolution and durable accounting for Aloy Run budgets."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from pori import BUDGET_EXHAUSTION_CODES, BudgetLedger, ExecutionBudget

from .models import Run
from .tenancy import OrganizationPolicy

BUDGET_STOP_REASONS = BUDGET_EXHAUSTION_CODES


class RunBudgetLimits(BaseModel):
    """Frozen ceilings copied onto a Run before it enters the worker queue."""

    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(ge=1)
    max_tool_calls: int = Field(ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, ge=0)
    timeout_seconds: int = Field(ge=1)

    def execution_budget(self) -> ExecutionBudget:
        return ExecutionBudget(
            max_steps=self.max_steps,
            max_tool_calls=self.max_tool_calls,
            max_tokens=self.max_tokens,
            max_cost_usd=self.max_cost_usd,
            max_duration_seconds=float(self.timeout_seconds),
        )


def _bounded_optional_int(
    requested: Any,
    policy_limit: int | None,
) -> int | None:
    requested_limit = int(requested) if requested is not None else None
    if requested_limit is None:
        return policy_limit
    if policy_limit is None:
        return requested_limit
    return min(requested_limit, policy_limit)


def _bounded_optional_float(
    requested: Any,
    policy_limit: float | None,
) -> float | None:
    requested_limit = float(requested) if requested is not None else None
    if requested_limit is None:
        return policy_limit
    if policy_limit is None:
        return requested_limit
    return min(requested_limit, policy_limit)


def resolve_run_budget(
    policy: OrganizationPolicy,
    requested: Mapping[str, Any] | None = None,
    *,
    default_max_steps: int | None = None,
) -> RunBudgetLimits:
    """Clamp requested ceilings to the organization-owned policy."""
    values = requested or {}
    requested_steps = values.get("max_steps")
    if requested_steps is None:
        requested_steps = default_max_steps or policy.max_steps_per_run
    requested_timeout = values.get("timeout_seconds")
    if requested_timeout is None:
        requested_timeout = policy.run_timeout_seconds
    requested_tool_calls = values.get("max_tool_calls")
    if requested_tool_calls is None:
        requested_tool_calls = policy.max_tool_calls_per_run
    return RunBudgetLimits(
        max_steps=min(int(requested_steps), policy.max_steps_per_run),
        max_tool_calls=min(
            int(requested_tool_calls),
            policy.max_tool_calls_per_run,
        ),
        max_tokens=_bounded_optional_int(
            values.get("max_tokens"),
            policy.max_tokens_per_run,
        ),
        max_cost_usd=_bounded_optional_float(
            values.get("max_cost_usd"),
            policy.max_cost_usd_per_run,
        ),
        timeout_seconds=min(
            int(requested_timeout),
            policy.run_timeout_seconds,
        ),
    )


def limits_for_run(run: Run) -> RunBudgetLimits:
    return RunBudgetLimits(
        max_steps=run.max_steps,
        max_tool_calls=run.max_tool_calls,
        max_tokens=run.max_tokens,
        max_cost_usd=run.max_cost_usd,
        timeout_seconds=run.timeout_seconds,
    )


def narrow_budget_to_parent(
    limits: RunBudgetLimits,
    parent: Run,
) -> RunBudgetLimits:
    """A child Run may never receive broader ceilings than its parent."""
    parent_limits = limits_for_run(parent)
    return RunBudgetLimits(
        max_steps=min(limits.max_steps, parent_limits.max_steps),
        max_tool_calls=min(
            limits.max_tool_calls,
            parent_limits.max_tool_calls,
        ),
        max_tokens=_bounded_optional_int(
            limits.max_tokens,
            parent_limits.max_tokens,
        ),
        max_cost_usd=_bounded_optional_float(
            limits.max_cost_usd,
            parent_limits.max_cost_usd,
        ),
        timeout_seconds=min(
            limits.timeout_seconds,
            parent_limits.timeout_seconds,
        ),
    )


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, str):
        try:
            return _as_utc(datetime.fromisoformat(value))
        except ValueError:
            return None
    return None


def elapsed_run_seconds(run: Run, *, now: datetime | None = None) -> float:
    """Return cumulative active execution time, excluding queue/user wait.

    ``runs.started_at`` is historical audit data and spans blocked user waits.
    Budget time is therefore checkpointed in ``progress.budget_usage`` and the
    current worker attempt contributes only from its claim/accounting marker.
    """
    progress = run.progress or {}
    persisted = progress.get("budget_usage")
    usage = persisted if isinstance(persisted, dict) else {}
    accounted = max(0.0, float(usage.get("duration_seconds_used") or 0.0))
    if run.status != "running":
        return accounted
    attempt_started = _timestamp(progress.get("budget_attempt_started_at"))
    accounted_at = _timestamp(progress.get("budget_accounted_at"))
    reference = accounted_at or attempt_started
    if attempt_started is not None and reference is not None:
        reference = max(reference, attempt_started)
    if reference is None:
        reference = _as_utc(run.started_at) if run.started_at is not None else None
    if reference is None:
        return accounted
    resolved_now = now or datetime.now(timezone.utc)
    return accounted + max(0.0, (resolved_now - reference).total_seconds())


def remaining_run_seconds(run: Run, *, now: datetime | None = None) -> float:
    return max(0.0, float(run.timeout_seconds) - elapsed_run_seconds(run, now=now))


def budget_ledger_for_run(
    run: Run,
    *,
    now: datetime | None = None,
) -> BudgetLedger:
    """Restore completed-attempt usage while preserving the total wall clock."""
    progress = run.progress or {}
    persisted = progress.get("budget_usage")
    usage = dict(persisted) if isinstance(persisted, dict) else {}
    usage["steps_used"] = max(
        int(usage.get("steps_used") or 0),
        int(progress.get("n_steps") or 0),
        int(run.steps_taken or 0),
    )
    usage["duration_seconds_used"] = elapsed_run_seconds(run, now=now)
    return BudgetLedger(limits_for_run(run).execution_budget(), initial_usage=usage)


__all__ = [
    "BUDGET_STOP_REASONS",
    "RunBudgetLimits",
    "budget_ledger_for_run",
    "elapsed_run_seconds",
    "limits_for_run",
    "narrow_budget_to_parent",
    "remaining_run_seconds",
    "resolve_run_budget",
]
