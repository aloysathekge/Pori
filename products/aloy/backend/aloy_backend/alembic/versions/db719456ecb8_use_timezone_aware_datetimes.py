"""use_timezone_aware_datetimes

Revision ID: db719456ecb8
Revises: 6336e9ee7ff3
Create Date: 2026-03-25 12:41:46.438382

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "db719456ecb8"
down_revision: Union[str, Sequence[str], None] = "6336e9ee7ff3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    for table, columns in (
        ("conversations", ("created_at", "updated_at")),
        ("messages", ("created_at",)),
        ("runs", ("created_at",)),
    ):
        with op.batch_alter_table(table) as batch_op:
            for column in columns:
                batch_op.alter_column(
                    column,
                    existing_type=sa.DateTime(),
                    type_=sa.DateTime(timezone=True),
                    existing_nullable=False,
                )


def downgrade() -> None:
    """Downgrade schema."""
    for table, columns in (
        ("runs", ("created_at",)),
        ("messages", ("created_at",)),
        ("conversations", ("updated_at", "created_at")),
    ):
        with op.batch_alter_table(table) as batch_op:
            for column in columns:
                batch_op.alter_column(
                    column,
                    existing_type=sa.DateTime(timezone=True),
                    type_=sa.DateTime(),
                    existing_nullable=False,
                )
