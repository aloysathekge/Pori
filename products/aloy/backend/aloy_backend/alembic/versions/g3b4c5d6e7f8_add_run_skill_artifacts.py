"""add run skill, artifact, and runtime event summaries

Revision ID: g3b4c5d6e7f8
Revises: f2a3b4c5d6e7
Create Date: 2026-06-24
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "g3b4c5d6e7f8"
down_revision: Union[str, Sequence[str], None] = "f2a3b4c5d6e7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("selected_skills", sa.JSON(), nullable=True))
    op.add_column("runs", sa.Column("artifacts", sa.JSON(), nullable=True))
    op.add_column("runs", sa.Column("runtime_events", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("runs", "runtime_events")
    op.drop_column("runs", "artifacts")
    op.drop_column("runs", "selected_skills")
