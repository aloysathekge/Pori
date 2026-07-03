"""add skill readiness metadata

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
Create Date: 2026-06-24
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision: str = "f2a3b4c5d6e7"
down_revision: Union[str, Sequence[str], None] = "e1f2a3b4c5d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "skill_definitions",
        sa.Column(
            "provenance",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default="organization",
        ),
    )
    op.add_column(
        "skill_definitions",
        sa.Column(
            "trust_level",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default="organization",
        ),
    )
    op.add_column(
        "skill_definitions",
        sa.Column("required_commands", sa.JSON(), nullable=False, server_default="[]"),
    )
    op.add_column(
        "skill_definitions",
        sa.Column(
            "setup_help",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default="",
        ),
    )
    op.add_column(
        "skill_definitions",
        sa.Column("readiness_warnings", sa.JSON(), nullable=False, server_default="[]"),
    )


def downgrade() -> None:
    op.drop_column("skill_definitions", "readiness_warnings")
    op.drop_column("skill_definitions", "setup_help")
    op.drop_column("skill_definitions", "required_commands")
    op.drop_column("skill_definitions", "trust_level")
    op.drop_column("skill_definitions", "provenance")
