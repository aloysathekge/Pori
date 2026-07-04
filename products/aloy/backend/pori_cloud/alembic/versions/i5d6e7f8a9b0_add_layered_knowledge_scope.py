"""add layered knowledge scope (org/team/personal) — the Aloy moat

Revision ID: i5d6e7f8a9b0
Revises: h4c5d6e7f8a9
Create Date: 2026-07-04
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "i5d6e7f8a9b0"
down_revision: Union[str, Sequence[str], None] = "h4c5d6e7f8a9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Existing rows are personal knowledge; org/team populate later.
    op.add_column(
        "knowledge_entries",
        sa.Column(
            "scope_level",
            sa.String(),
            nullable=False,
            server_default="personal",
        ),
    )
    op.add_column(
        "knowledge_entries",
        sa.Column("team_id", sa.String(), nullable=True),
    )
    op.create_index(
        "ix_knowledge_entries_scope_level",
        "knowledge_entries",
        ["scope_level"],
    )
    op.create_index(
        "ix_knowledge_entries_team_id",
        "knowledge_entries",
        ["team_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_knowledge_entries_team_id", table_name="knowledge_entries")
    op.drop_index("ix_knowledge_entries_scope_level", table_name="knowledge_entries")
    op.drop_column("knowledge_entries", "team_id")
    op.drop_column("knowledge_entries", "scope_level")
