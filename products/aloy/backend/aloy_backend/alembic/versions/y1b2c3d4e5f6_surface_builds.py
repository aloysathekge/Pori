"""add durable isolated Surface build records

Revision ID: y1b2c3d4e5f6
Revises: x0a1b2c3d4e5
Create Date: 2026-07-16
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "y1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "x0a1b2c3d4e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "surface_builds",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("revision_id", sa.String(), nullable=False),
        sa.Column("creator_run_id", sa.String(), nullable=True),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("request_fingerprint", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("source_checksum", sa.String(), nullable=False),
        sa.Column("toolchain_version", sa.String(), nullable=False),
        sa.Column("validation_result", sa.JSON(), nullable=False),
        sa.Column("diagnostics", sa.JSON(), nullable=False),
        sa.Column("build_log", sa.String(), nullable=False),
        sa.Column("bundle_key", sa.String(), nullable=True),
        sa.Column("bundle_sha256", sa.String(), nullable=True),
        sa.Column("bundle_size_bytes", sa.Integer(), nullable=False),
        sa.Column("preview_artifacts", sa.JSON(), nullable=False),
        sa.Column("resource_metrics", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["surface_projects.id"]),
        sa.ForeignKeyConstraint(["revision_id"], ["surface_revisions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id",
            "idempotency_key",
            name="uq_surface_build_idempotency",
        ),
    )
    for column in (
        "organization_id",
        "user_id",
        "event_id",
        "project_id",
        "revision_id",
        "creator_run_id",
        "status",
        "source_checksum",
        "bundle_sha256",
    ):
        op.create_index(
            f"ix_surface_builds_{column}",
            "surface_builds",
            [column],
        )


def downgrade() -> None:
    op.drop_table("surface_builds")
