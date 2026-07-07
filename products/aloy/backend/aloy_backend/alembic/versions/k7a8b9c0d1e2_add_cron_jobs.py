"""add cron jobs (recurring runs, marathon Phase 3)

Revision ID: k7a8b9c0d1e2
Revises: j6e7f8a9b0c1
Create Date: 2026-07-06
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "k7a8b9c0d1e2"
down_revision: Union[str, Sequence[str], None] = "j6e7f8a9b0c1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "cron_jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("task", sa.String(), nullable=False),
        sa.Column("schedule", sa.String(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("max_steps", sa.Integer(), nullable=False, server_default="15"),
        sa.Column("conversation_id", sa.String(), nullable=True),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_run_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_cron_jobs_organization_id", "cron_jobs", ["organization_id"])
    op.create_index("ix_cron_jobs_user_id", "cron_jobs", ["user_id"])
    op.create_index("ix_cron_jobs_conversation_id", "cron_jobs", ["conversation_id"])
    op.create_index("ix_cron_jobs_next_run_at", "cron_jobs", ["next_run_at"])


def downgrade() -> None:
    op.drop_index("ix_cron_jobs_next_run_at", table_name="cron_jobs")
    op.drop_index("ix_cron_jobs_conversation_id", table_name="cron_jobs")
    op.drop_index("ix_cron_jobs_user_id", table_name="cron_jobs")
    op.drop_index("ix_cron_jobs_organization_id", table_name="cron_jobs")
    op.drop_table("cron_jobs")
