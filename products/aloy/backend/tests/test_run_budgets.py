from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect

from aloy_backend.models import Run
from aloy_backend.run_budgets import (
    budget_ledger_for_run,
    elapsed_run_seconds,
    narrow_budget_to_parent,
    remaining_run_seconds,
    resolve_run_budget,
)
from aloy_backend.tenancy import OrganizationPolicy


def _run(**overrides) -> Run:
    values = {
        "organization_id": "org-budget",
        "user_id": "alice",
        "event_id": "evt-budget",
        "agent_id": "agent-budget",
        "session_id": "session-budget",
        "task": "bounded work",
    }
    values.update(overrides)
    return Run(**values)


def test_run_budget_is_clamped_and_child_cannot_broaden_parent():
    policy = OrganizationPolicy(
        max_steps_per_run=20,
        max_tool_calls_per_run=30,
        max_tokens_per_run=50_000,
        max_cost_usd_per_run=2.5,
        run_timeout_seconds=600,
    )
    requested = resolve_run_budget(
        policy,
        {
            "max_steps": 99,
            "max_tool_calls": 99,
            "max_tokens": 100_000,
            "max_cost_usd": 10,
            "timeout_seconds": 1_000,
        },
    )

    assert requested.model_dump() == {
        "max_steps": 20,
        "max_tool_calls": 30,
        "max_tokens": 50_000,
        "max_cost_usd": 2.5,
        "timeout_seconds": 600,
    }

    parent = _run(
        max_steps=8,
        max_tool_calls=12,
        max_tokens=9_000,
        max_cost_usd=0.75,
        timeout_seconds=120,
    )
    child = narrow_budget_to_parent(requested, parent)
    assert child.model_dump() == {
        "max_steps": 8,
        "max_tool_calls": 12,
        "max_tokens": 9_000,
        "max_cost_usd": 0.75,
        "timeout_seconds": 120,
    }


def test_duration_budget_counts_active_attempts_but_not_waiting_time():
    now = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
    waiting = _run(
        status="pending",
        timeout_seconds=100,
        started_at=now - timedelta(days=2),
        progress={"budget_usage": {"duration_seconds_used": 20.0}},
    )
    assert elapsed_run_seconds(waiting, now=now) == 20.0
    assert remaining_run_seconds(waiting, now=now) == 80.0

    running = _run(
        status="running",
        timeout_seconds=100,
        progress={
            "budget_usage": {"duration_seconds_used": 20.0},
            "budget_attempt_started_at": (now - timedelta(seconds=5)).isoformat(),
        },
    )
    assert elapsed_run_seconds(running, now=now) == 25.0
    assert remaining_run_seconds(running, now=now) == 75.0
    assert (
        budget_ledger_for_run(running, now=now).snapshot()["duration_seconds_used"]
        == 25.0
    )


def test_run_budget_migration_round_trip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'run-budgets.db'}")
    metadata = sa.MetaData()
    sa.Table("runs", metadata, sa.Column("id", sa.String(), primary_key=True))
    metadata.create_all(engine)

    with engine.begin() as connection:
        migration = importlib.import_module(
            "aloy_backend.alembic.versions.k3b4c5d6e7f8_run_execution_budgets"
        )
        original_op = migration.op
        migration.op = Operations(MigrationContext.configure(connection))
        try:
            migration.upgrade()
            columns = {
                column["name"] for column in inspect(connection).get_columns("runs")
            }
            assert {"max_tool_calls", "max_tokens", "max_cost_usd"} <= columns
            migration.downgrade()
            assert {
                column["name"] for column in inspect(connection).get_columns("runs")
            } == {"id"}
            migration.upgrade()
            columns = {
                column["name"] for column in inspect(connection).get_columns("runs")
            }
            assert {"max_tool_calls", "max_tokens", "max_cost_usd"} <= columns
        finally:
            migration.op = original_op
    engine.dispose()
