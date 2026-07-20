"""Add durable Conversation summary provenance boundaries.

Revision ID: l4c5d6e7f8a9
Revises: k3b4c5d6e7f8
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "l4c5d6e7f8a9"
down_revision: str | None = "k3b4c5d6e7f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("context_artifacts") as batch:
        batch.add_column(
            sa.Column(
                "summary_version", sa.Integer(), nullable=False, server_default="0"
            )
        )
        batch.add_column(
            sa.Column("source_start_message_id", sa.String(), nullable=True)
        )
        batch.add_column(sa.Column("source_end_message_id", sa.String(), nullable=True))
        batch.add_column(
            sa.Column("source_started_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch.add_column(
            sa.Column("source_ended_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch.add_column(
            sa.Column(
                "source_message_count", sa.Integer(), nullable=False, server_default="0"
            )
        )
        batch.add_column(
            sa.Column(
                "content_fingerprint", sa.String(), nullable=False, server_default=""
            )
        )
        batch.create_index("ix_context_artifacts_summary_version", ["summary_version"])
        batch.create_index(
            "ix_context_artifacts_source_start_message_id",
            ["source_start_message_id"],
        )
        batch.create_index(
            "ix_context_artifacts_source_end_message_id", ["source_end_message_id"]
        )
        batch.create_index(
            "uq_context_summary_version",
            ["conversation_id", "artifact_type", "summary_version"],
            unique=True,
            sqlite_where=sa.text("summary_version > 0"),
            postgresql_where=sa.text("summary_version > 0"),
        )


def downgrade() -> None:
    with op.batch_alter_table("context_artifacts") as batch:
        batch.drop_index("uq_context_summary_version")
        batch.drop_index("ix_context_artifacts_source_end_message_id")
        batch.drop_index("ix_context_artifacts_source_start_message_id")
        batch.drop_index("ix_context_artifacts_summary_version")
        batch.drop_column("content_fingerprint")
        batch.drop_column("source_message_count")
        batch.drop_column("source_ended_at")
        batch.drop_column("source_started_at")
        batch.drop_column("source_end_message_id")
        batch.drop_column("source_start_message_id")
        batch.drop_column("summary_version")
