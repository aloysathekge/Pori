"""Harden durable Run timeline sequencing and replay.

Revision ID: r0c1d2e3f4a5
Revises: q9b0c1d2e3f4
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "r0c1d2e3f4a5"
down_revision: str | None = "q9b0c1d2e3f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "run_timeline_cursors",
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("last_sequence", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.add_column(
        "run_timeline_events",
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "run_timeline_events",
        sa.Column("source_event_key", sa.String(), nullable=True),
    )

    bind = op.get_bind()
    events = sa.table(
        "run_timeline_events",
        sa.column("id", sa.String()),
        sa.column("run_id", sa.String()),
        sa.column("sequence", sa.Integer()),
        sa.column("source_event_key", sa.String()),
    )
    cursors = sa.table(
        "run_timeline_cursors",
        sa.column("run_id", sa.String()),
        sa.column("last_sequence", sa.Integer()),
    )
    rows = bind.execute(sa.select(events.c.id)).all()
    for (event_id,) in rows:
        bind.execute(
            events.update()
            .where(events.c.id == event_id)
            .values(source_event_key=f"legacy:{event_id}")
        )
    maxima = bind.execute(
        sa.select(events.c.run_id, sa.func.max(events.c.sequence)).group_by(
            events.c.run_id
        )
    ).all()
    if maxima:
        op.bulk_insert(
            cursors,
            [
                {"run_id": run_id, "last_sequence": int(sequence)}
                for run_id, sequence in maxima
            ],
        )

    with op.batch_alter_table("run_timeline_events") as batch:
        batch.alter_column(
            "source_event_key", existing_type=sa.String(), nullable=False
        )
        batch.create_unique_constraint(
            "uq_run_timeline_source_event", ["run_id", "source_event_key"]
        )
        batch.create_index(
            "ix_run_timeline_events_source_event_key", ["source_event_key"]
        )


def downgrade() -> None:
    with op.batch_alter_table("run_timeline_events") as batch:
        batch.drop_index("ix_run_timeline_events_source_event_key")
        batch.drop_constraint("uq_run_timeline_source_event", type_="unique")
        batch.drop_column("source_event_key")
        batch.drop_column("schema_version")
    op.drop_table("run_timeline_cursors")
