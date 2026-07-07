"""unify memory records

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-06-08
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, Sequence[str], None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "knowledge_entries", sa.Column("organization_id", sa.String(), nullable=True)
    )
    op.add_column(
        "knowledge_entries", sa.Column("agent_id", sa.String(), nullable=True)
    )
    op.add_column(
        "knowledge_entries", sa.Column("session_id", sa.String(), nullable=True)
    )
    op.add_column(
        "knowledge_entries",
        sa.Column("kind", sa.String(), nullable=False, server_default="semantic"),
    )
    op.add_column(
        "knowledge_entries",
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1"),
    )
    op.add_column(
        "knowledge_entries",
        sa.Column(
            "sensitivity", sa.String(), nullable=False, server_default="internal"
        ),
    )
    op.add_column(
        "knowledge_entries", sa.Column("provenance", sa.JSON(), nullable=True)
    )
    op.add_column("knowledge_entries", sa.Column("retention", sa.JSON(), nullable=True))
    op.add_column(
        "knowledge_entries", sa.Column("conflict_key", sa.String(), nullable=True)
    )
    op.add_column(
        "knowledge_entries",
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
    )
    op.add_column(
        "knowledge_entries", sa.Column("superseded_by", sa.String(), nullable=True)
    )
    op.add_column(
        "knowledge_entries",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "knowledge_entries",
        sa.Column("event_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "knowledge_entries",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.execute("UPDATE knowledge_entries SET organization_id = 'user:' || user_id")
    op.execute("UPDATE knowledge_entries SET updated_at = created_at")
    with op.batch_alter_table("knowledge_entries") as batch_op:
        batch_op.alter_column(
            "organization_id", existing_type=sa.String(), nullable=False
        )
        batch_op.alter_column(
            "updated_at",
            existing_type=sa.DateTime(timezone=True),
            nullable=False,
        )

    op.create_index(
        "ix_knowledge_entries_organization_id", "knowledge_entries", ["organization_id"]
    )
    op.create_index("ix_knowledge_entries_agent_id", "knowledge_entries", ["agent_id"])
    op.create_index(
        "ix_knowledge_entries_session_id", "knowledge_entries", ["session_id"]
    )
    op.create_index(
        "ix_knowledge_entries_conflict_key", "knowledge_entries", ["conflict_key"]
    )
    op.create_index(
        "ix_knowledge_entries_scope_status",
        "knowledge_entries",
        ["organization_id", "user_id", "agent_id", "session_id", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_knowledge_entries_scope_status", table_name="knowledge_entries")
    op.drop_index("ix_knowledge_entries_conflict_key", table_name="knowledge_entries")
    op.drop_index("ix_knowledge_entries_session_id", table_name="knowledge_entries")
    op.drop_index("ix_knowledge_entries_agent_id", table_name="knowledge_entries")
    op.drop_index(
        "ix_knowledge_entries_organization_id", table_name="knowledge_entries"
    )
    for column in (
        "deleted_at",
        "event_at",
        "updated_at",
        "superseded_by",
        "status",
        "conflict_key",
        "retention",
        "provenance",
        "sensitivity",
        "confidence",
        "kind",
        "session_id",
        "agent_id",
        "organization_id",
    ):
        op.drop_column("knowledge_entries", column)
