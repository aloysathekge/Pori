"""oauth connection scope (personal + org-shared connections)

Revision ID: o1e2f3a4b5c6
Revises: n0d1e2f3a4b5
Create Date: 2026-07-08
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "o1e2f3a4b5c6"
down_revision: Union[str, Sequence[str], None] = "n0d1e2f3a4b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "oauth_connections",
        sa.Column("scope", sa.String(), nullable=False, server_default="user"),
    )
    op.add_column(
        "oauth_connections", sa.Column("created_by", sa.String(), nullable=True)
    )
    op.create_index("ix_oauth_connections_scope", "oauth_connections", ["scope"])
    op.add_column(
        "oauth_flow_states",
        sa.Column("scope", sa.String(), nullable=False, server_default="user"),
    )


def downgrade() -> None:
    op.drop_column("oauth_flow_states", "scope")
    op.drop_index("ix_oauth_connections_scope", table_name="oauth_connections")
    op.drop_column("oauth_connections", "created_by")
    op.drop_column("oauth_connections", "scope")
