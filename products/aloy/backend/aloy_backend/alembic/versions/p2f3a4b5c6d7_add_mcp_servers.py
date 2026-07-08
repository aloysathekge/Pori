"""add mcp servers (personal + org-shared MCP)

Revision ID: p2f3a4b5c6d7
Revises: o1e2f3a4b5c6
Create Date: 2026-07-08
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "p2f3a4b5c6d7"
down_revision: Union[str, Sequence[str], None] = "o1e2f3a4b5c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "mcp_servers",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("scope", sa.String(), nullable=False, server_default="user"),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("transport", sa.String(), nullable=False, server_default="http"),
        sa.Column("url", sa.String(), nullable=False, server_default=""),
        sa.Column("auth_kind", sa.String(), nullable=False, server_default="none"),
        sa.Column("static_secret_enc", sa.String(), nullable=True),
        sa.Column("oauth_connection_id", sa.String(), nullable=True),
        sa.Column("tools_include", sa.JSON(), nullable=True),
        sa.Column("tools_exclude", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "organization_id", "user_id", "name", name="uq_mcp_server_name"
        ),
    )
    op.create_index(
        "ix_mcp_servers_organization_id", "mcp_servers", ["organization_id"]
    )
    op.create_index("ix_mcp_servers_user_id", "mcp_servers", ["user_id"])
    op.create_index("ix_mcp_servers_scope", "mcp_servers", ["scope"])


def downgrade() -> None:
    op.drop_index("ix_mcp_servers_scope", table_name="mcp_servers")
    op.drop_index("ix_mcp_servers_user_id", table_name="mcp_servers")
    op.drop_index("ix_mcp_servers_organization_id", table_name="mcp_servers")
    op.drop_table("mcp_servers")
