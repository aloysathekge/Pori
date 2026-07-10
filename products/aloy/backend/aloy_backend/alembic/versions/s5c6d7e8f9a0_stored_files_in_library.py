"""stored_files.in_library — the user file library flag

Revision ID: s5c6d7e8f9a0
Revises: r4b5c6d7e8f9
Create Date: 2026-07-10
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "s5c6d7e8f9a0"
down_revision: Union[str, Sequence[str], None] = "r4b5c6d7e8f9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "stored_files",
        sa.Column(
            "in_library", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
    )
    op.create_index("ix_stored_files_in_library", "stored_files", ["in_library"])


def downgrade() -> None:
    op.drop_index("ix_stored_files_in_library", table_name="stored_files")
    op.drop_column("stored_files", "in_library")
