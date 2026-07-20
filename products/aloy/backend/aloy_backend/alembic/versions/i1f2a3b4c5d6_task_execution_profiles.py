"""Add explicit purpose profile selection to Event Tasks.

Revision ID: i1f2a3b4c5d6
Revises: h0e1f2a3b4c5
Create Date: 2026-07-19
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "i1f2a3b4c5d6"
down_revision = "h0e1f2a3b4c5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("tasks") as batch_op:
        batch_op.add_column(
            sa.Column(
                "execution_profile",
                sa.String(),
                nullable=False,
                server_default="general",
            )
        )
        batch_op.create_index(
            "ix_tasks_execution_profile", ["execution_profile"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("tasks") as batch_op:
        batch_op.drop_index("ix_tasks_execution_profile")
        batch_op.drop_column("execution_profile")
