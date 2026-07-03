"""add_core_memory_blocks_and_knowledge_entries

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-03-31

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "core_memory_blocks",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False, index=True),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False, server_default=""),
        sa.Column("char_limit", sa.Integer(), nullable=False, server_default="2000"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "user_id",
            "label",
            name="uq_core_memory_blocks_user_label",
        ),
    )

    op.create_table(
        "knowledge_entries",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False, index=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("importance", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("source", sa.String(), nullable=False, server_default="agent"),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("knowledge_entries")
    op.drop_table("core_memory_blocks")
