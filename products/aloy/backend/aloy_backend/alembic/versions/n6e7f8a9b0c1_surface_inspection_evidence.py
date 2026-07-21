"""Persist immutable trusted Surface inspection receipts and evidence.

Revision ID: n6e7f8a9b0c1
Revises: m5d6e7f8a9b0
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "n6e7f8a9b0c1"
down_revision: str | None = "m5d6e7f8a9b0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "surface_inspections",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("build_id", sa.String(), nullable=False),
        sa.Column("revision_id", sa.String(), nullable=False),
        sa.Column("bundle_key", sa.String(), nullable=False),
        sa.Column("bundle_sha256", sa.String(), nullable=False),
        sa.Column("inspection_kind", sa.String(), nullable=False),
        sa.Column("inspector_version", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("receipt_sha256", sa.String(), nullable=False),
        sa.Column("policy_versions", sa.JSON(), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("timings", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["surface_projects.id"]),
        sa.ForeignKeyConstraint(["build_id"], ["surface_builds.id"]),
        sa.ForeignKeyConstraint(["revision_id"], ["surface_revisions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in (
        "organization_id",
        "user_id",
        "event_id",
        "project_id",
        "build_id",
        "revision_id",
        "bundle_key",
        "bundle_sha256",
        "inspection_kind",
        "status",
        "receipt_sha256",
    ):
        op.create_index(
            f"ix_surface_inspections_{column}", "surface_inspections", [column]
        )

    op.create_table(
        "surface_evidence_artifacts",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("inspection_id", sa.String(), nullable=False),
        sa.Column("build_id", sa.String(), nullable=False),
        sa.Column("revision_id", sa.String(), nullable=False),
        sa.Column("bundle_key", sa.String(), nullable=False),
        sa.Column("bundle_sha256", sa.String(), nullable=False),
        sa.Column("artifact_kind", sa.String(), nullable=False),
        sa.Column("storage_key", sa.String(), nullable=False),
        sa.Column("content_type", sa.String(), nullable=False),
        sa.Column("content_sha256", sa.String(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["surface_projects.id"]),
        sa.ForeignKeyConstraint(["inspection_id"], ["surface_inspections.id"]),
        sa.ForeignKeyConstraint(["build_id"], ["surface_builds.id"]),
        sa.ForeignKeyConstraint(["revision_id"], ["surface_revisions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in (
        "organization_id",
        "user_id",
        "event_id",
        "project_id",
        "inspection_id",
        "build_id",
        "revision_id",
        "bundle_key",
        "bundle_sha256",
        "artifact_kind",
        "storage_key",
        "content_sha256",
    ):
        op.create_index(
            f"ix_surface_evidence_artifacts_{column}",
            "surface_evidence_artifacts",
            [column],
        )


def downgrade() -> None:
    op.drop_table("surface_evidence_artifacts")
    op.drop_table("surface_inspections")
