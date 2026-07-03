"""add_team_configs

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-27 14:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "team_configs",
        sa.Column("id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("user_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("mode", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("members", sa.JSON(), nullable=False),
        sa.Column("max_delegation_steps", sa.Integer(), nullable=False),
        sa.Column("max_concurrent_members", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_team_configs_user_id", "team_configs", ["user_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_team_configs_user_id", "team_configs")
    op.drop_table("team_configs")
