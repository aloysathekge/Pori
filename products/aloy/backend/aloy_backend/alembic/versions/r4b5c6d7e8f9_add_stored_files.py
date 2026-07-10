"""add stored_files (object-store pointer rows)

Revision ID: r4b5c6d7e8f9
Revises: q3a4b5c6d7e8
Create Date: 2026-07-10
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "r4b5c6d7e8f9"
down_revision: Union[str, Sequence[str], None] = "q3a4b5c6d7e8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "stored_files",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("conversation_id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=True),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("content_type", sa.String(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(), nullable=False),
        sa.Column("storage_key", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_stored_files_organization_id", "stored_files", ["organization_id"]
    )
    op.create_index("ix_stored_files_user_id", "stored_files", ["user_id"])
    op.create_index(
        "ix_stored_files_conversation_id", "stored_files", ["conversation_id"]
    )
    op.create_index("ix_stored_files_run_id", "stored_files", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_stored_files_run_id", table_name="stored_files")
    op.drop_index("ix_stored_files_conversation_id", table_name="stored_files")
    op.drop_index("ix_stored_files_user_id", table_name="stored_files")
    op.drop_index("ix_stored_files_organization_id", table_name="stored_files")
    op.drop_table("stored_files")
