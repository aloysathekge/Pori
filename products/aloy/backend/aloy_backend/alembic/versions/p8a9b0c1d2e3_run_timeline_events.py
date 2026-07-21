"""Add durable Run timeline events.

Revision ID: p8a9b0c1d2e3
Revises: o7f8a9b0c1d2
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "p8a9b0c1d2e3"
down_revision: str | None = "o7f8a9b0c1d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "run_timeline_events",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("conversation_id", sa.String(), nullable=True),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("public_payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "sequence", name="uq_run_timeline_sequence"),
    )
    for column in (
        "organization_id",
        "user_id",
        "event_id",
        "conversation_id",
        "run_id",
        "sequence",
        "kind",
    ):
        op.create_index(
            f"ix_run_timeline_events_{column}",
            "run_timeline_events",
            [column],
        )


def downgrade() -> None:
    op.drop_table("run_timeline_events")
