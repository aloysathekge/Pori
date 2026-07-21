"""Add durable plan state to Event Tasks.

Revision ID: q9b0c1d2e3f4
Revises: p8a9b0c1d2e3
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "q9b0c1d2e3f4"
down_revision: str | None = "p8a9b0c1d2e3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "tasks",
        sa.Column("plan", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
    )
    op.add_column(
        "tasks",
        sa.Column("plan_version", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "tasks",
        sa.Column(
            "current_activity",
            sa.String(),
            nullable=False,
            server_default=sa.text("''"),
        ),
    )


def downgrade() -> None:
    op.drop_column("tasks", "current_activity")
    op.drop_column("tasks", "plan_version")
    op.drop_column("tasks", "plan")
