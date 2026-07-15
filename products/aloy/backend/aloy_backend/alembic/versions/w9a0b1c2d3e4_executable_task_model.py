"""add the executable Task model and link Runs

Revision ID: w9a0b1c2d3e4
Revises: v8f9a0b1c2d3
Create Date: 2026-07-15
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "w9a0b1c2d3e4"
down_revision: Union[str, Sequence[str], None] = "v8f9a0b1c2d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("tasks") as batch:
        batch.add_column(
            sa.Column("origin_conversation_id", sa.String(), nullable=True)
        )
        batch.add_column(sa.Column("instructions", sa.String(), nullable=True))
        batch.add_column(sa.Column("definition_of_done", sa.String(), nullable=True))
        batch.add_column(sa.Column("priority", sa.String(), nullable=True))
        batch.add_column(sa.Column("due_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(sa.Column("execution_mode", sa.String(), nullable=True))
        batch.add_column(sa.Column("assigned_agent_id", sa.String(), nullable=True))
        batch.add_column(sa.Column("current_run_id", sa.String(), nullable=True))
        batch.add_column(sa.Column("result_summary", sa.String(), nullable=True))
        batch.add_column(sa.Column("blocker", sa.String(), nullable=True))
        batch.add_column(sa.Column("budget_policy", sa.JSON(), nullable=True))

    connection = op.get_bind()
    tasks = sa.table(
        "tasks",
        sa.column("id", sa.String()),
        sa.column("event_id", sa.String()),
        sa.column("status", sa.String()),
        sa.column("origin_conversation_id", sa.String()),
        sa.column("instructions", sa.String()),
        sa.column("definition_of_done", sa.String()),
        sa.column("priority", sa.String()),
        sa.column("execution_mode", sa.String()),
        sa.column("result_summary", sa.String()),
        sa.column("blocker", sa.String()),
        sa.column("budget_policy", sa.JSON()),
    )
    events = sa.table(
        "events",
        sa.column("id", sa.String()),
        sa.column("primary_conversation_id", sa.String()),
    )
    connection.execute(
        tasks.update().values(
            instructions="",
            definition_of_done="",
            priority="normal",
            execution_mode="manual",
            result_summary="",
            blocker="",
            budget_policy={},
        )
    )
    connection.execute(
        tasks.update()
        .where(~tasks.c.status.in_(["open", "done"]))
        .values(status="open")
    )
    rows = connection.execute(
        sa.select(tasks.c.id, events.c.primary_conversation_id).select_from(
            tasks.join(events, tasks.c.event_id == events.c.id)
        )
    ).all()
    for task_id, conversation_id in rows:
        if conversation_id:
            connection.execute(
                tasks.update()
                .where(tasks.c.id == task_id)
                .values(origin_conversation_id=conversation_id)
            )

    with op.batch_alter_table("tasks") as batch:
        batch.alter_column("instructions", existing_type=sa.String(), nullable=False)
        batch.alter_column(
            "definition_of_done", existing_type=sa.String(), nullable=False
        )
        batch.alter_column("priority", existing_type=sa.String(), nullable=False)
        batch.alter_column("execution_mode", existing_type=sa.String(), nullable=False)
        batch.alter_column("result_summary", existing_type=sa.String(), nullable=False)
        batch.alter_column("blocker", existing_type=sa.String(), nullable=False)
        batch.alter_column("budget_policy", existing_type=sa.JSON(), nullable=False)
        batch.create_index(
            "ix_tasks_origin_conversation_id", ["origin_conversation_id"]
        )
        batch.create_index("ix_tasks_priority", ["priority"])
        batch.create_index("ix_tasks_execution_mode", ["execution_mode"])
        batch.create_index("ix_tasks_assigned_agent_id", ["assigned_agent_id"])
        batch.create_index("ix_tasks_current_run_id", ["current_run_id"])

    with op.batch_alter_table("runs") as batch:
        batch.add_column(sa.Column("task_id", sa.String(), nullable=True))
        batch.create_index("ix_runs_task_id", ["task_id"])


def downgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.drop_index("ix_runs_task_id")
        batch.drop_column("task_id")
    with op.batch_alter_table("tasks") as batch:
        batch.drop_index("ix_tasks_current_run_id")
        batch.drop_index("ix_tasks_assigned_agent_id")
        batch.drop_index("ix_tasks_execution_mode")
        batch.drop_index("ix_tasks_priority")
        batch.drop_index("ix_tasks_origin_conversation_id")
        batch.drop_column("budget_policy")
        batch.drop_column("blocker")
        batch.drop_column("result_summary")
        batch.drop_column("current_run_id")
        batch.drop_column("assigned_agent_id")
        batch.drop_column("execution_mode")
        batch.drop_column("due_at")
        batch.drop_column("priority")
        batch.drop_column("definition_of_done")
        batch.drop_column("instructions")
        batch.drop_column("origin_conversation_id")
