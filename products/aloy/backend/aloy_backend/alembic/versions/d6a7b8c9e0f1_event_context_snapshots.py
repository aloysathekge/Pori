"""Add immutable Event context snapshots and evidence-linked briefs."""

import sqlalchemy as sa
from alembic import op

revision = "d6a7b8c9e0f1"
down_revision = "c5f6a7b8d9e0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "event_context_snapshots",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), sa.ForeignKey("events.id"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.String(), nullable=False),
        sa.Column("fingerprint", sa.String(), nullable=False),
        sa.Column("readiness", sa.String(), nullable=False),
        sa.Column("provider_cache_allowed", sa.Boolean(), nullable=False),
        sa.Column("pack", sa.JSON(), nullable=False),
        sa.Column("evidence_refs", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("event_id", "version", name="uq_event_context_version"),
        sa.UniqueConstraint(
            "event_id", "fingerprint", name="uq_event_context_fingerprint"
        ),
    )
    for column in (
        "organization_id",
        "user_id",
        "event_id",
        "version",
        "fingerprint",
        "readiness",
    ):
        op.create_index(
            f"ix_event_context_snapshots_{column}",
            "event_context_snapshots",
            [column],
        )
    op.create_table(
        "event_briefs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), sa.ForeignKey("events.id"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column(
            "source_context_snapshot_id",
            sa.String(),
            sa.ForeignKey("event_context_snapshots.id"),
            nullable=False,
        ),
        sa.Column("creator_run_id", sa.String(), nullable=True),
        sa.Column("fingerprint", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("evidence_refs", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("event_id", "version", name="uq_event_brief_version"),
        sa.UniqueConstraint(
            "event_id", "fingerprint", name="uq_event_brief_fingerprint"
        ),
    )
    for column in (
        "organization_id",
        "user_id",
        "event_id",
        "version",
        "status",
        "source_context_snapshot_id",
        "creator_run_id",
        "fingerprint",
    ):
        op.create_index(f"ix_event_briefs_{column}", "event_briefs", [column])


def downgrade() -> None:
    op.drop_table("event_briefs")
    op.drop_table("event_context_snapshots")
