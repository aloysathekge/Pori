"""add oauth connections + flow state (account connections)

Revision ID: n0d1e2f3a4b5
Revises: m9c0d1e2f3a4
Create Date: 2026-07-08
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "n0d1e2f3a4b5"
down_revision: Union[str, Sequence[str], None] = "m9c0d1e2f3a4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "oauth_connections",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("access_token_enc", sa.String(), nullable=False),
        sa.Column("refresh_token_enc", sa.String(), nullable=True),
        sa.Column("scopes", sa.JSON(), nullable=False),
        sa.Column("account_email", sa.String(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "organization_id", "user_id", "provider", name="uq_oauth_conn"
        ),
    )
    op.create_index(
        "ix_oauth_connections_organization_id", "oauth_connections", ["organization_id"]
    )
    op.create_index("ix_oauth_connections_user_id", "oauth_connections", ["user_id"])
    op.create_index("ix_oauth_connections_provider", "oauth_connections", ["provider"])

    op.create_table(
        "oauth_flow_states",
        sa.Column("state", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("pkce_verifier", sa.String(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_oauth_flow_states_organization_id", "oauth_flow_states", ["organization_id"]
    )
    op.create_index("ix_oauth_flow_states_user_id", "oauth_flow_states", ["user_id"])


def downgrade() -> None:
    op.drop_table("oauth_flow_states")
    op.drop_index("ix_oauth_connections_provider", table_name="oauth_connections")
    op.drop_index("ix_oauth_connections_user_id", table_name="oauth_connections")
    op.drop_index(
        "ix_oauth_connections_organization_id", table_name="oauth_connections"
    )
    op.drop_table("oauth_connections")
