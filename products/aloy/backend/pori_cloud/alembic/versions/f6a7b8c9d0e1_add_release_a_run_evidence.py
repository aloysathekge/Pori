"""add release A run identity and evidence

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-06-20
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f6a7b8c9d0e1"
down_revision: Union[str, Sequence[str], None] = "e5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("organization_id", sa.String(), nullable=True))
    op.add_column("runs", sa.Column("agent_id", sa.String(), nullable=True))
    op.add_column("runs", sa.Column("session_id", sa.String(), nullable=True))
    op.add_column("runs", sa.Column("prompt_fingerprint", sa.String(), nullable=True))
    op.add_column(
        "runs", sa.Column("tool_surface_fingerprint", sa.String(), nullable=True)
    )
    op.add_column("runs", sa.Column("execution_receipts", sa.JSON(), nullable=True))
    op.execute("UPDATE runs SET organization_id = 'user:' || user_id")
    op.execute("UPDATE runs SET agent_id = 'default_agent'")
    op.execute("UPDATE runs SET session_id = COALESCE(conversation_id, id)")
    with op.batch_alter_table("runs") as batch_op:
        for column in ("organization_id", "agent_id", "session_id"):
            batch_op.alter_column(column, existing_type=sa.String(), nullable=False)
    for column in ("organization_id", "agent_id", "session_id"):
        op.create_index(f"ix_runs_{column}", "runs", [column])

    op.add_column(
        "trace_records", sa.Column("organization_id", sa.String(), nullable=True)
    )
    op.execute("UPDATE trace_records SET organization_id = 'user:' || user_id")
    with op.batch_alter_table("trace_records") as batch_op:
        batch_op.alter_column(
            "organization_id", existing_type=sa.String(), nullable=False
        )
    op.create_index(
        "ix_trace_records_organization_id", "trace_records", ["organization_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_trace_records_organization_id", table_name="trace_records")
    op.drop_column("trace_records", "organization_id")
    for column in ("session_id", "agent_id", "organization_id"):
        op.drop_index(f"ix_runs_{column}", table_name="runs")
    for column in (
        "execution_receipts",
        "tool_surface_fingerprint",
        "prompt_fingerprint",
        "session_id",
        "agent_id",
        "organization_id",
    ):
        op.drop_column("runs", column)
