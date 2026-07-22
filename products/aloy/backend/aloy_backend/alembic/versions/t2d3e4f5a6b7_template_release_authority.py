"""Add staged template catalog metadata and operator audit receipts.

Revision ID: t2d3e4f5a6b7
Revises: s1d2e3f4a5b6
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "t2d3e4f5a6b7"
down_revision: str | None = "s1d2e3f4a5b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "event_template_releases",
        sa.Column(
            "catalog_snapshot",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )
    op.create_table(
        "event_template_operator_receipts",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("intent_id", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("request_fingerprint", sa.String(), nullable=False),
        sa.Column("template_id", sa.String(), nullable=False),
        sa.Column("release_id", sa.String(), nullable=False),
        sa.Column("reason", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("receipt", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "idempotency_key",
            name="uq_event_template_operator_receipt_request",
        ),
        sa.UniqueConstraint(
            "intent_id",
            name="uq_event_template_operator_receipt_intent",
        ),
    )
    for column in (
        "organization_id",
        "user_id",
        "action",
        "intent_id",
        "request_fingerprint",
        "template_id",
        "release_id",
        "status",
    ):
        op.create_index(
            f"ix_event_template_operator_receipts_{column}",
            "event_template_operator_receipts",
            [column],
        )


def downgrade() -> None:
    op.drop_table("event_template_operator_receipts")
    op.drop_column("event_template_releases", "catalog_snapshot")
