"""add run plan column; drop unused runtime_events

Revision ID: h4c5d6e7f8a9
Revises: g3b4c5d6e7f8
Create Date: 2026-06-30
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "h4c5d6e7f8a9"
down_revision: Union[str, Sequence[str], None] = "g3b4c5d6e7f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("plan", sa.JSON(), nullable=True))
    # runtime_events had no producer in Pori core and was always empty.
    op.drop_column("runs", "runtime_events")


def downgrade() -> None:
    op.add_column("runs", sa.Column("runtime_events", sa.JSON(), nullable=True))
    op.drop_column("runs", "plan")
