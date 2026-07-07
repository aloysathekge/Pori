"""add run event logs (read-only replay viewer)

Revision ID: m9c0d1e2f3a4
Revises: l8b9c0d1e2f3
Create Date: 2026-07-07
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "m9c0d1e2f3a4"
down_revision: Union[str, Sequence[str], None] = "l8b9c0d1e2f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "run_event_logs",
        sa.Column("run_id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("conversation_id", sa.String(), nullable=True),
        sa.Column("events", sa.JSON(), nullable=False),
        sa.Column("event_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_run_event_logs_organization_id", "run_event_logs", ["organization_id"]
    )
    op.create_index("ix_run_event_logs_user_id", "run_event_logs", ["user_id"])
    op.create_index(
        "ix_run_event_logs_conversation_id", "run_event_logs", ["conversation_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_run_event_logs_conversation_id", table_name="run_event_logs")
    op.drop_index("ix_run_event_logs_user_id", table_name="run_event_logs")
    op.drop_index("ix_run_event_logs_organization_id", table_name="run_event_logs")
    op.drop_table("run_event_logs")
