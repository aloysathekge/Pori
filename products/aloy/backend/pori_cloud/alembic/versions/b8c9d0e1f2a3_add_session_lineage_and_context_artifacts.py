"""add session lineage and context artifacts

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-06-22
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b8c9d0e1f2a3"
down_revision: Union[str, Sequence[str], None] = "a7b8c9d0e1f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("conversations") as batch:
        batch.add_column(
            sa.Column("parent_conversation_id", sa.String(), nullable=True)
        )
        batch.add_column(
            sa.Column("branched_from_message_id", sa.String(), nullable=True)
        )
        batch.create_index(
            "ix_conversations_parent_conversation_id",
            ["parent_conversation_id"],
        )
    op.create_table(
        "context_artifacts",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("conversation_id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=True),
        sa.Column("artifact_type", sa.String(), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("source_message_ids", sa.JSON(), nullable=False),
        sa.Column("diagnostics", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    for column in (
        "organization_id",
        "user_id",
        "conversation_id",
        "run_id",
        "artifact_type",
    ):
        op.create_index(f"ix_context_artifacts_{column}", "context_artifacts", [column])


def downgrade() -> None:
    op.drop_table("context_artifacts")
    with op.batch_alter_table("conversations") as batch:
        batch.drop_index("ix_conversations_parent_conversation_id")
        batch.drop_column("branched_from_message_id")
        batch.drop_column("parent_conversation_id")
