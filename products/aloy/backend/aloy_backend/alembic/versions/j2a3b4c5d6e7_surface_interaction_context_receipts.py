"""Persist exact context-read proof for Surface-triggered Runs.

Revision ID: j2a3b4c5d6e7
Revises: i1f2a3b4c5d6
Create Date: 2026-07-20
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "j2a3b4c5d6e7"
down_revision = "i1f2a3b4c5d6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("surface_interactions") as batch_op:
        batch_op.add_column(
            sa.Column("context_read_run_id", sa.String(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("context_read_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.create_index(
            "ix_surface_interactions_context_read_run_id",
            ["context_read_run_id"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("surface_interactions") as batch_op:
        batch_op.drop_index("ix_surface_interactions_context_read_run_id")
        batch_op.drop_column("context_read_at")
        batch_op.drop_column("context_read_run_id")
