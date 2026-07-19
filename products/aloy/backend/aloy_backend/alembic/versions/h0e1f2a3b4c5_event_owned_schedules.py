"""Make recurring schedules Event-owned and link their execution history.

Revision ID: h0e1f2a3b4c5
Revises: g9d0e1f2b3c4
Create Date: 2026-07-19
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "h0e1f2a3b4c5"
down_revision = "g9d0e1f2b3c4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("cron_jobs") as batch_op:
        batch_op.add_column(sa.Column("event_id", sa.String(), nullable=True))
        batch_op.add_column(
            sa.Column("timezone", sa.String(), nullable=False, server_default="UTC")
        )
        batch_op.add_column(
            sa.Column(
                "authority",
                sa.String(),
                nullable=False,
                server_default="report_only",
            )
        )
        batch_op.add_column(
            sa.Column(
                "notification_mode",
                sa.String(),
                nullable=False,
                server_default="attention",
            )
        )
        batch_op.add_column(
            sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_cron_jobs_event_id_events", "events", ["event_id"], ["id"]
        )
    op.create_index("ix_cron_jobs_event_id", "cron_jobs", ["event_id"])
    op.create_index("ix_cron_jobs_authority", "cron_jobs", ["authority"])
    op.create_index(
        "ix_cron_jobs_notification_mode", "cron_jobs", ["notification_mode"]
    )
    op.create_index("ix_cron_jobs_deleted_at", "cron_jobs", ["deleted_at"])

    with op.batch_alter_table("runs") as batch_op:
        batch_op.add_column(sa.Column("cron_job_id", sa.String(), nullable=True))
        batch_op.create_foreign_key(
            "fk_runs_cron_job_id_cron_jobs", "cron_jobs", ["cron_job_id"], ["id"]
        )
    op.create_index("ix_runs_cron_job_id", "runs", ["cron_job_id"])


def downgrade() -> None:
    op.drop_index("ix_runs_cron_job_id", table_name="runs")
    with op.batch_alter_table("runs") as batch_op:
        batch_op.drop_constraint(
            "fk_runs_cron_job_id_cron_jobs", type_="foreignkey"
        )
        batch_op.drop_column("cron_job_id")

    op.drop_index("ix_cron_jobs_deleted_at", table_name="cron_jobs")
    op.drop_index("ix_cron_jobs_notification_mode", table_name="cron_jobs")
    op.drop_index("ix_cron_jobs_authority", table_name="cron_jobs")
    op.drop_index("ix_cron_jobs_event_id", table_name="cron_jobs")
    with op.batch_alter_table("cron_jobs") as batch_op:
        batch_op.drop_constraint("fk_cron_jobs_event_id_events", type_="foreignkey")
        batch_op.drop_column("notification_mode")
        batch_op.drop_column("authority")
        batch_op.drop_column("timezone")
        batch_op.drop_column("deleted_at")
        batch_op.drop_column("event_id")
