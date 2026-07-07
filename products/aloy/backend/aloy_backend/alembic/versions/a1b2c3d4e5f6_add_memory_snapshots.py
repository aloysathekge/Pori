"""add_memory_snapshots

Revision ID: a1b2c3d4e5f6
Revises: f4a1b2c3d4e5
Create Date: 2026-03-27 13:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "f4a1b2c3d4e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "memory_snapshots",
        sa.Column("namespace", sa.String(), nullable=False),
        sa.Column(
            "payload",
            JSONB().with_variant(sa.JSON(), "sqlite"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("namespace"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("memory_snapshots")
