"""Add durable Event setup drafts, context items, and connection grants."""

import sqlalchemy as sa
from alembic import op

revision = "b4e5f6a7c8d9"
down_revision = "a3d4e5f6b7c8"
branch_labels = None
depends_on = None


def _indexes(table: str, columns: tuple[str, ...]) -> None:
    for column in columns:
        op.create_index(f"ix_{table}_{column}", table, [column])


def upgrade() -> None:
    op.create_table(
        "event_setup_drafts",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("mode", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("created_event_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["created_event_id"], ["events.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("created_event_id", name="uq_event_setup_draft_event"),
    )
    _indexes(
        "event_setup_drafts",
        ("organization_id", "user_id", "mode", "status", "created_event_id"),
    )

    op.create_table(
        "event_setup_context_items",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("draft_id", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("source_url", sa.String(), nullable=True),
        sa.Column("connection_id", sa.String(), nullable=True),
        sa.Column("access_scope", sa.JSON(), nullable=False),
        sa.Column("storage_key", sa.String(), nullable=True),
        sa.Column("content_type", sa.String(), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(), nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["draft_id"], ["event_setup_drafts.id"]),
        sa.ForeignKeyConstraint(["connection_id"], ["oauth_connections.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    _indexes(
        "event_setup_context_items",
        ("organization_id", "user_id", "draft_id", "kind", "status", "connection_id"),
    )

    op.create_table(
        "event_connection_grants",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("connection_id", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("access_scope", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.ForeignKeyConstraint(["connection_id"], ["oauth_connections.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "event_id", "connection_id", name="uq_event_connection_grant"
        ),
    )
    _indexes(
        "event_connection_grants",
        (
            "organization_id",
            "user_id",
            "event_id",
            "connection_id",
            "provider",
            "status",
        ),
    )


def downgrade() -> None:
    op.drop_table("event_connection_grants")
    op.drop_table("event_setup_context_items")
    op.drop_table("event_setup_drafts")
