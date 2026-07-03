"""initial_run_table

Revision ID: a577558e49f6
Revises: 
Create Date: 2026-03-25 12:14:18.102357

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a577558e49f6"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "runs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("task", sa.String(), nullable=False),
        sa.Column("max_steps", sa.Integer(), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("steps_taken", sa.Integer(), nullable=False),
        sa.Column("final_answer", sa.String(), nullable=True),
        sa.Column("reasoning", sa.String(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_runs_user_id", "runs", ["user_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_runs_user_id", table_name="runs")
    op.drop_table("runs")
