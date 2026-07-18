"""Freeze developer-owned model-role assignments onto Runs."""

import sqlalchemy as sa
from alembic import op

revision = "f8c9d0e1a2b3"
down_revision = "e7b8c9d0f1a2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column("model_assignment", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("runs", "model_assignment")
