"""Persist the complete host-owned Run execution budget.

Revision ID: k3b4c5d6e7f8
Revises: j2a3b4c5d6e7
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "k3b4c5d6e7f8"
down_revision: str | None = "j2a3b4c5d6e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.add_column(
            sa.Column(
                "max_tool_calls",
                sa.Integer(),
                nullable=False,
                server_default="100",
            )
        )
        batch.add_column(sa.Column("max_tokens", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("max_cost_usd", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.drop_column("max_cost_usd")
        batch.drop_column("max_tokens")
        batch.drop_column("max_tool_calls")
