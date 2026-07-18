"""Add the append-only Surface command-attempt outcome ledger.

Revision ID: g9d0e1f2b3c4
Revises: f8c9d0e1a2b3
Create Date: 2026-07-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "g9d0e1f2b3c4"
down_revision = "f8c9d0e1a2b3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "surface_command_attempts",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("build_id", sa.String(), nullable=False),
        sa.Column("code_revision_id", sa.String(), nullable=False),
        sa.Column("interaction_id", sa.String(), nullable=True),
        sa.Column("method", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("interaction_class", sa.String(), nullable=False),
        sa.Column("component_id", sa.String(), nullable=False),
        sa.Column("actor_id", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("request_fingerprint", sa.String(), nullable=False),
        sa.Column("base_data_revision", sa.Integer(), nullable=False),
        sa.Column("observed_data_revision", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error_code", sa.String(), nullable=True),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("http_status", sa.Integer(), nullable=False),
        sa.Column("retryable", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["build_id"], ["surface_builds.id"]),
        sa.ForeignKeyConstraint(["code_revision_id"], ["surface_revisions.id"]),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.ForeignKeyConstraint(["interaction_id"], ["surface_interactions.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["surface_projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in (
        "organization_id",
        "user_id",
        "event_id",
        "project_id",
        "build_id",
        "code_revision_id",
        "interaction_id",
        "method",
        "name",
        "interaction_class",
        "actor_id",
        "idempotency_key",
        "request_fingerprint",
        "status",
        "error_code",
    ):
        op.create_index(
            f"ix_surface_command_attempts_{column}",
            "surface_command_attempts",
            [column],
        )


def downgrade() -> None:
    op.drop_table("surface_command_attempts")
