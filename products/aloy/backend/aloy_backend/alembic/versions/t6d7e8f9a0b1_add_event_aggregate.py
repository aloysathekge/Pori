"""add the Aloy V1 Event aggregate

Revision ID: t6d7e8f9a0b1
Revises: s5c6d7e8f9a0
Create Date: 2026-07-14
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "t6d7e8f9a0b1"
down_revision: Union[str, Sequence[str], None] = "s5c6d7e8f9a0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _index(table: str, column: str) -> None:
    op.create_index(f"ix_{table}_{column}", table, [column])


def _add_event_reference(table: str, *, nullable: bool = True) -> None:
    with op.batch_alter_table(table) as batch:
        batch.add_column(sa.Column("event_id", sa.String(), nullable=nullable))
        batch.create_foreign_key(
            f"fk_{table}_event_id_events", "events", ["event_id"], ["id"]
        )
        batch.create_index(f"ix_{table}_event_id", ["event_id"])


def _drop_event_reference(table: str) -> None:
    with op.batch_alter_table(table) as batch:
        batch.drop_index(f"ix_{table}_event_id")
        batch.drop_constraint(f"fk_{table}_event_id_events", type_="foreignkey")
        batch.drop_column("event_id")


def upgrade() -> None:
    op.create_table(
        "events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("lifecycle", sa.String(), nullable=False),
        sa.Column("phase", sa.String(), nullable=False),
        sa.Column("summary", sa.String(), nullable=False),
        sa.Column("is_life", sa.Boolean(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    for column in ("organization_id", "user_id", "type", "lifecycle", "is_life"):
        _index("events", column)
    op.create_index(
        "uq_events_life_per_user",
        "events",
        ["organization_id", "user_id"],
        unique=True,
        sqlite_where=sa.text("is_life = 1"),
        postgresql_where=sa.text("is_life = true"),
    )

    op.create_table(
        "tasks",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), sa.ForeignKey("events.id"), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("order", sa.Integer(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    for column in ("organization_id", "user_id", "event_id", "status"):
        _index("tasks", column)

    op.create_table(
        "proposals",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), sa.ForeignKey("events.id"), nullable=False),
        sa.Column("origin_session_id", sa.String(), nullable=True),
        sa.Column("origin_run_id", sa.String(), nullable=True),
        sa.Column("tool", sa.String(), nullable=False),
        sa.Column("args", sa.JSON(), nullable=False),
        sa.Column("tool_schema_fingerprint", sa.String(), nullable=False),
        sa.Column("reason", sa.String(), nullable=False),
        sa.Column("impact", sa.String(), nullable=False),
        sa.Column("risk", sa.String(), nullable=False),
        sa.Column("routing", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("safe_default", sa.JSON(), nullable=True),
        sa.Column("decided_by", sa.String(), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("receipt", sa.JSON(), nullable=True),
        sa.Column("execution_attempt_id", sa.String(), nullable=True),
        sa.Column("provider_operation_id", sa.String(), nullable=True),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    for column in (
        "organization_id",
        "user_id",
        "event_id",
        "origin_session_id",
        "origin_run_id",
        "tool",
        "risk",
        "routing",
        "status",
        "decided_by",
        "execution_attempt_id",
        "provider_operation_id",
    ):
        _index("proposals", column)

    op.create_table(
        "event_trail_entries",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), sa.ForeignKey("events.id"), nullable=False),
        sa.Column("actor_id", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("summary", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=True),
        sa.Column("proposal_id", sa.String(), nullable=True),
        sa.Column("task_id", sa.String(), nullable=True),
        sa.Column("evidence_refs", sa.JSON(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    for column in (
        "organization_id",
        "user_id",
        "event_id",
        "actor_id",
        "kind",
        "run_id",
        "proposal_id",
        "task_id",
    ):
        _index("event_trail_entries", column)

    for table in (
        "conversations",
        "runs",
        "stored_files",
        "context_artifacts",
        "trace_records",
        "knowledge_entries",
        "run_event_logs",
    ):
        _add_event_reference(table)
    with op.batch_alter_table("stored_files") as batch:
        batch.add_column(sa.Column("origin_session_id", sa.String(), nullable=True))
        batch.create_index("ix_stored_files_origin_session_id", ["origin_session_id"])

    connection = op.get_bind()
    sources = [
        sa.select(sa.column("organization_id"), sa.column("user_id")).select_from(
            sa.table(table)
        )
        for table in (
            "conversations",
            "runs",
            "stored_files",
            "context_artifacts",
            "trace_records",
            "knowledge_entries",
            "run_event_logs",
            "core_memory_blocks",
        )
    ]
    identities = connection.execute(sa.union(*sources)).all()
    events = sa.table(
        "events",
        sa.column("id", sa.String()),
        sa.column("organization_id", sa.String()),
        sa.column("user_id", sa.String()),
        sa.column("type", sa.String()),
        sa.column("title", sa.String()),
        sa.column("lifecycle", sa.String()),
        sa.column("phase", sa.String()),
        sa.column("summary", sa.String()),
        sa.column("is_life", sa.Boolean()),
        sa.column("metadata", sa.JSON()),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )
    now = datetime.now(timezone.utc)
    life_ids: dict[tuple[str, str], str] = {}
    for organization_id, user_id in identities:
        if not organization_id or not user_id:
            continue
        event_id = f"evt_{uuid.uuid4().hex}"
        life_ids[(organization_id, user_id)] = event_id
        connection.execute(
            events.insert().values(
                id=event_id,
                organization_id=organization_id,
                user_id=user_id,
                type="life",
                title="Life",
                lifecycle="active",
                phase="",
                summary="",
                is_life=True,
                metadata={},
                created_at=now,
                updated_at=now,
            )
        )

    for table_name in (
        "conversations",
        "runs",
        "stored_files",
        "context_artifacts",
        "trace_records",
        "run_event_logs",
    ):
        table = sa.table(
            table_name,
            sa.column("organization_id"),
            sa.column("user_id"),
            sa.column("event_id"),
        )
        for (organization_id, user_id), event_id in life_ids.items():
            connection.execute(
                table.update()
                .where(
                    table.c.organization_id == organization_id,
                    table.c.user_id == user_id,
                )
                .values(event_id=event_id)
            )
    stored_files = sa.table(
        "stored_files", sa.column("origin_session_id"), sa.column("conversation_id")
    )
    connection.execute(
        stored_files.update().values(origin_session_id=stored_files.c.conversation_id)
    )

    for table in (
        "conversations",
        "runs",
        "stored_files",
        "context_artifacts",
        "trace_records",
        "run_event_logs",
    ):
        with op.batch_alter_table(table) as batch:
            batch.alter_column("event_id", existing_type=sa.String(), nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("stored_files") as batch:
        batch.drop_index("ix_stored_files_origin_session_id")
        batch.drop_column("origin_session_id")
    for table in (
        "run_event_logs",
        "knowledge_entries",
        "trace_records",
        "context_artifacts",
        "stored_files",
        "runs",
        "conversations",
    ):
        _drop_event_reference(table)
    op.drop_table("event_trail_entries")
    op.drop_table("proposals")
    op.drop_table("tasks")
    op.drop_table("events")
