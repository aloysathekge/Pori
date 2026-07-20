"""Add durable Proposal reconciliation scheduling.

Revision ID: m5d6e7f8a9b0
Revises: l4c5d6e7f8a9
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "m5d6e7f8a9b0"
down_revision: str | None = "l4c5d6e7f8a9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("proposals") as batch:
        batch.add_column(
            sa.Column(
                "reconciliation_attempts",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )
        batch.add_column(
            sa.Column(
                "reconciliation_checked_at",
                sa.DateTime(timezone=True),
                nullable=True,
            )
        )
        batch.add_column(
            sa.Column(
                "reconciliation_next_at",
                sa.DateTime(timezone=True),
                nullable=True,
            )
        )
        batch.create_index(
            "ix_proposals_reconciliation_next_at", ["reconciliation_next_at"]
        )


def downgrade() -> None:
    with op.batch_alter_table("proposals") as batch:
        batch.drop_index("ix_proposals_reconciliation_next_at")
        batch.drop_column("reconciliation_next_at")
        batch.drop_column("reconciliation_checked_at")
        batch.drop_column("reconciliation_attempts")
