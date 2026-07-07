"""add run progress checkpoint (resume-not-restart, marathon Phase 2)

Revision ID: j6e7f8a9b0c1
Revises: i5d6e7f8a9b0
Create Date: 2026-07-06
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "j6e7f8a9b0c1"
down_revision: Union[str, Sequence[str], None] = "i5d6e7f8a9b0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Per-step loop checkpoint: lets a re-claimed run resume from its last
    # step instead of re-executing the whole task from scratch.
    op.add_column("runs", sa.Column("progress", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("runs", "progress")
