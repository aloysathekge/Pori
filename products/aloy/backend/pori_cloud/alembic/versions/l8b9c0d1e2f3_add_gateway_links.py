"""add gateway links + pairing codes (Telegram gateway slice)

Revision ID: l8b9c0d1e2f3
Revises: k7a8b9c0d1e2
Create Date: 2026-07-07
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "l8b9c0d1e2f3"
down_revision: Union[str, Sequence[str], None] = "k7a8b9c0d1e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "gateway_links",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("platform", sa.String(), nullable=False),
        sa.Column("chat_id", sa.String(), nullable=False),
        sa.Column("chat_title", sa.String(), nullable=True),
        sa.Column("conversation_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_gateway_links_organization_id", "gateway_links", ["organization_id"]
    )
    op.create_index("ix_gateway_links_user_id", "gateway_links", ["user_id"])
    op.create_index("ix_gateway_links_platform", "gateway_links", ["platform"])
    op.create_index("ix_gateway_links_chat_id", "gateway_links", ["chat_id"])
    op.create_index(
        "ix_gateway_links_conversation_id", "gateway_links", ["conversation_id"]
    )

    op.create_table(
        "gateway_pairing_codes",
        sa.Column("code", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("platform", sa.String(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_gateway_pairing_codes_organization_id",
        "gateway_pairing_codes",
        ["organization_id"],
    )
    op.create_index(
        "ix_gateway_pairing_codes_user_id", "gateway_pairing_codes", ["user_id"]
    )


def downgrade() -> None:
    op.drop_table("gateway_pairing_codes")
    op.drop_table("gateway_links")
