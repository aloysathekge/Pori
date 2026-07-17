"""Add durable leasing and provenance state for Event context ingestion."""

import sqlalchemy as sa
from alembic import op

revision = "c5f6a7b8d9e0"
down_revision = "b4e5f6a7c8d9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "event_setup_context_items",
        sa.Column("event_id", sa.String(), nullable=True),
    )
    op.add_column(
        "event_setup_context_items",
        sa.Column(
            "sensitivity", sa.String(), nullable=False, server_default="internal"
        ),
    )
    op.add_column(
        "event_setup_context_items",
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "event_setup_context_items",
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
    )
    op.add_column(
        "event_setup_context_items",
        sa.Column("lease_owner", sa.String(), nullable=True),
    )
    op.add_column(
        "event_setup_context_items",
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "event_setup_context_items",
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "event_setup_context_items",
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "event_setup_context_items",
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "event_setup_context_items",
        sa.Column("knowledge_entry_id", sa.String(), nullable=True),
    )
    op.create_index(
        "ix_event_setup_context_items_event_id",
        "event_setup_context_items",
        ["event_id"],
    )
    op.create_index(
        "ix_event_setup_context_items_sensitivity",
        "event_setup_context_items",
        ["sensitivity"],
    )
    op.create_index(
        "ix_event_setup_context_items_lease_owner",
        "event_setup_context_items",
        ["lease_owner"],
    )
    op.create_index(
        "ix_event_setup_context_items_knowledge_entry_id",
        "event_setup_context_items",
        ["knowledge_entry_id"],
    )
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        op.create_foreign_key(
            "fk_event_setup_context_items_event_id",
            "event_setup_context_items",
            "events",
            ["event_id"],
            ["id"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        op.drop_constraint(
            "fk_event_setup_context_items_event_id",
            "event_setup_context_items",
            type_="foreignkey",
        )
    for index in (
        "ix_event_setup_context_items_knowledge_entry_id",
        "ix_event_setup_context_items_lease_owner",
        "ix_event_setup_context_items_sensitivity",
        "ix_event_setup_context_items_event_id",
    ):
        op.drop_index(index, table_name="event_setup_context_items")
    for column in (
        "knowledge_entry_id",
        "ingested_at",
        "retrieved_at",
        "next_attempt_at",
        "lease_expires_at",
        "lease_owner",
        "max_attempts",
        "attempt_count",
        "sensitivity",
        "event_id",
    ):
        op.drop_column("event_setup_context_items", column)
