"""Add purpose-scoped Event bootstrap Run persistence."""

import sqlalchemy as sa
from alembic import op

revision = "e7b8c9d0f1a2"
down_revision = "d6a7b8c9e0f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "event_context_snapshots",
        sa.Column(
            "evidence_payload",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )
    # Batch mode preserves SQLite support while remaining a normal ALTER on
    # databases that can add foreign keys directly.
    with op.batch_alter_table("runs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "run_kind",
                sa.String(),
                nullable=False,
                server_default="agent",
            )
        )
        batch_op.add_column(
            sa.Column(
                "context_snapshot_id",
                sa.String(),
                nullable=True,
            )
        )
        batch_op.add_column(sa.Column("run_profile", sa.JSON(), nullable=True))
        batch_op.create_foreign_key(
            "fk_runs_context_snapshot_id_event_context_snapshots",
            "event_context_snapshots",
            ["context_snapshot_id"],
            ["id"],
        )
    op.create_index("ix_runs_run_kind", "runs", ["run_kind"])
    op.create_index("ix_runs_context_snapshot_id", "runs", ["context_snapshot_id"])
    op.create_index(
        "uq_runs_event_context_kind",
        "runs",
        ["event_id", "run_kind", "context_snapshot_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_runs_event_context_kind", table_name="runs")
    op.drop_index("ix_runs_context_snapshot_id", table_name="runs")
    op.drop_index("ix_runs_run_kind", table_name="runs")
    with op.batch_alter_table("runs") as batch_op:
        batch_op.drop_constraint(
            "fk_runs_context_snapshot_id_event_context_snapshots",
            type_="foreignkey",
        )
        batch_op.drop_column("run_profile")
        batch_op.drop_column("context_snapshot_id")
        batch_op.drop_column("run_kind")
    op.drop_column("event_context_snapshots", "evidence_payload")
