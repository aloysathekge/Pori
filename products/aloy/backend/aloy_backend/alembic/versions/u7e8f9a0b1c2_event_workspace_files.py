"""allow Event artifacts without a Session

Revision ID: u7e8f9a0b1c2
Revises: t6d7e8f9a0b1
Create Date: 2026-07-14
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "u7e8f9a0b1c2"
down_revision: Union[str, Sequence[str], None] = "t6d7e8f9a0b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("stored_files") as batch:
        batch.alter_column("conversation_id", existing_type=sa.String(), nullable=True)


def downgrade() -> None:
    stored_files = sa.table(
        "stored_files",
        sa.column("conversation_id", sa.String()),
        sa.column("origin_session_id", sa.String()),
        sa.column("event_id", sa.String()),
    )
    op.execute(
        stored_files.update()
        .where(stored_files.c.conversation_id.is_(None))
        .values(
            conversation_id=sa.func.coalesce(
                stored_files.c.origin_session_id, stored_files.c.event_id
            )
        )
    )
    with op.batch_alter_table("stored_files") as batch:
        batch.alter_column("conversation_id", existing_type=sa.String(), nullable=False)
