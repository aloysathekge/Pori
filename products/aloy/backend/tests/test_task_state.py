from __future__ import annotations

import asyncio
import importlib
from datetime import datetime, timezone

import sqlalchemy as sa
import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect
from sqlmodel import select

from aloy_backend.models import EventTrailEntry, Run, Task
from aloy_backend.task_state import (
    TaskStateError,
    claim_task,
    validate_task_transition,
)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        ("open", "queued"),
        ("queued", "in_progress"),
        ("in_progress", "blocked"),
        ("in_progress", "waiting_approval"),
        ("in_progress", "done"),
        ("blocked", "queued"),
        ("waiting_approval", "in_progress"),
        ("done", "open"),
        ("failed", "queued"),
        ("cancelled", "open"),
    ],
)
def test_legal_task_transitions(current, target):
    validate_task_transition(current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [("open", "in_progress"), ("queued", "done"), ("done", "queued")],
)
def test_illegal_task_transitions(current, target):
    with pytest.raises(TaskStateError, match="Illegal Task transition"):
        validate_task_transition(current, target)


async def _life(client) -> tuple[dict, dict, str]:
    first = (await client.post("/v1/conversations", json={"title": "First"})).json()
    second = (await client.post("/v1/conversations", json={"title": "Second"})).json()
    return first, second, first["event_id"]


async def test_task_contract_preserves_life_origin_and_rejects_cross_event(client):
    first, second, life_event_id = await _life(client)
    created = await client.post(
        f"/v1/events/{life_event_id}/tasks",
        json={
            "title": "Research startup roles",
            "instructions": "Find current US startup openings.",
            "definition_of_done": "A cited shortlist exists.",
            "priority": "high",
            "due_at": "2026-08-01T12:00:00Z",
            "origin_conversation_id": first["id"],
            "budget_policy": {"max_steps": 20, "max_tool_calls": 12},
        },
    )
    assert created.status_code == 201
    task = created.json()
    assert task["event_id"] == life_event_id
    assert task["origin_conversation_id"] == first["id"]
    assert task["instructions"] == "Find current US startup openings."
    assert task["definition_of_done"] == "A cited shortlist exists."
    assert task["priority"] == "high"
    assert task["due_at"] == "2026-08-01T12:00:00Z"
    assert task["execution_mode"] == "manual"
    assert task["budget_policy"] == {"max_steps": 20, "max_tool_calls": 12}
    assert task["current_run_id"] is None

    sibling = await client.post(
        f"/v1/events/{life_event_id}/tasks",
        json={"title": "Second origin", "origin_conversation_id": second["id"]},
    )
    assert sibling.status_code == 201
    assert sibling.json()["origin_conversation_id"] == second["id"]

    project = (await client.post("/v1/events", json={"title": "Career OS"})).json()
    denied = await client.post(
        f"/v1/events/{project['id']}/tasks",
        json={"title": "Leak", "origin_conversation_id": first["id"]},
    )
    assert denied.status_code == 409
    assert "must belong to its Event" in denied.json()["detail"]


async def test_illegal_task_transition_has_no_partial_state_or_trail(
    client, db_session_maker
):
    project = (await client.post("/v1/events", json={"title": "Transitions"})).json()
    task = (
        await client.post(
            f"/v1/events/{project['id']}/tasks", json={"title": "State machine"}
        )
    ).json()

    queued = await client.patch(
        f"/v1/events/{project['id']}/tasks/{task['id']}",
        json={"status": "queued"},
    )
    assert queued.status_code == 200
    illegal = await client.patch(
        f"/v1/events/{project['id']}/tasks/{task['id']}",
        json={"status": "done"},
    )
    assert illegal.status_code == 409
    assert illegal.json()["detail"] == "Illegal Task transition: queued -> done"

    async with db_session_maker() as session:
        stored = await session.get(Task, task["id"])
        entries = (
            (
                await session.execute(
                    select(EventTrailEntry).where(EventTrailEntry.task_id == task["id"])
                )
            )
            .scalars()
            .all()
        )
    assert stored is not None and stored.status == "queued"
    assert [entry.payload["action"] for entry in entries] == ["created", "updated"]


async def test_concurrent_task_claims_have_one_winner_and_one_trail(
    client, db_session_maker
):
    project = (await client.post("/v1/events", json={"title": "Claims"})).json()
    task = (
        await client.post(
            f"/v1/events/{project['id']}/tasks", json={"title": "Claim once"}
        )
    ).json()
    assert (
        await client.patch(
            f"/v1/events/{project['id']}/tasks/{task['id']}",
            json={"status": "queued"},
        )
    ).status_code == 200

    run_ids = ["run-claim-a", "run-claim-b"]
    async with db_session_maker() as session:
        session.add_all(
            [
                Run(
                    id=run_id,
                    organization_id="user:test-user",
                    user_id="test-user",
                    event_id=project["id"],
                    task_id=task["id"],
                    agent_id="default_agent",
                    session_id=project["conversation_id"],
                    conversation_id=project["conversation_id"],
                    task="Claim the Task",
                    status="pending",
                )
                for run_id in run_ids
            ]
        )
        await session.commit()

    async def attempt(run_id: str):
        async with db_session_maker() as session:
            return await claim_task(
                session,
                organization_id="user:test-user",
                user_id="test-user",
                event_id=project["id"],
                task_id=task["id"],
                run_id=run_id,
                actor_id="worker",
            )

    results = await asyncio.gather(*(attempt(run_id) for run_id in run_ids))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert winners[0].current_run_id in run_ids
    assert winners[0].status == "in_progress"

    async with db_session_maker() as session:
        stored = await session.get(Task, task["id"])
        claim_entries = (
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.task_id == task["id"],
                        EventTrailEntry.kind == "task_changed",
                    )
                )
            )
            .scalars()
            .all()
        )
    assert stored is not None
    assert stored.status == "in_progress"
    assert stored.current_run_id == winners[0].current_run_id
    assert [entry.payload["action"] for entry in claim_entries].count("claimed") == 1


def test_executable_task_migration_backfills_existing_rows(tmp_path):
    db_path = tmp_path / "r2-migration.db"
    engine = create_engine(f"sqlite:///{db_path}")
    metadata = sa.MetaData()
    sa.Table(
        "events",
        metadata,
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("primary_conversation_id", sa.String(), nullable=True),
    )
    sa.Table(
        "tasks",
        metadata,
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("order", sa.Integer(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    sa.Table("runs", metadata, sa.Column("id", sa.String(), primary_key=True))
    metadata.create_all(engine)
    now = datetime.now(timezone.utc)
    with engine.begin() as connection:
        connection.execute(
            metadata.tables["events"]
            .insert()
            .values(id="evt-old", primary_conversation_id="conv-old")
        )
        connection.execute(
            metadata.tables["tasks"]
            .insert()
            .values(
                id="task-old",
                organization_id="org-old",
                user_id="user-old",
                event_id="evt-old",
                title="Existing work",
                status="open",
                order=0,
                created_by="user-old",
                created_at=now,
                updated_at=now,
            )
        )
        migration = importlib.import_module(
            "aloy_backend.alembic.versions.w9a0b1c2d3e4_executable_task_model"
        )
        original_op = migration.op
        migration.op = Operations(MigrationContext.configure(connection))
        try:
            migration.upgrade()
        finally:
            migration.op = original_op

        row = connection.execute(sa.text("SELECT * FROM tasks")).mappings().one()
        assert row["title"] == "Existing work"
        assert row["status"] == "open"
        assert row["origin_conversation_id"] == "conv-old"
        assert row["priority"] == "normal"
        assert row["execution_mode"] == "manual"
        assert row["budget_policy"] == "{}"

    task_columns = {column["name"] for column in inspect(engine).get_columns("tasks")}
    run_columns = {column["name"] for column in inspect(engine).get_columns("runs")}
    assert {"instructions", "definition_of_done", "current_run_id"} <= task_columns
    assert "task_id" in run_columns
    engine.dispose()
