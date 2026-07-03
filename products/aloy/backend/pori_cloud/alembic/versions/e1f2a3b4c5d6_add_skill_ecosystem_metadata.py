"""add skill ecosystem metadata

Revision ID: e1f2a3b4c5d6
Revises: d0e1f2a3b4c5
Create Date: 2026-06-24
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision: str = "e1f2a3b4c5d6"
down_revision: Union[str, Sequence[str], None] = "d0e1f2a3b4c5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "skill_definitions",
        sa.Column(
            "category",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default="organization",
        ),
    )
    op.add_column(
        "skill_definitions",
        sa.Column(
            "author",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default="",
        ),
    )
    op.add_column(
        "skill_definitions",
        sa.Column(
            "license",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default="",
        ),
    )
    op.add_column(
        "skill_definitions",
        sa.Column("commands", sa.JSON(), nullable=False, server_default="[]"),
    )
    op.add_column(
        "skill_definitions",
        sa.Column(
            "argument_hint",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default="",
        ),
    )
    op.add_column(
        "skill_definitions",
        sa.Column(
            "source_url",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default="",
        ),
    )
    op.add_column(
        "skill_definitions",
        sa.Column(
            "install_command",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default="",
        ),
    )


def downgrade() -> None:
    op.drop_column("skill_definitions", "install_command")
    op.drop_column("skill_definitions", "source_url")
    op.drop_column("skill_definitions", "argument_hint")
    op.drop_column("skill_definitions", "commands")
    op.drop_column("skill_definitions", "license")
    op.drop_column("skill_definitions", "author")
    op.drop_column("skill_definitions", "category")
