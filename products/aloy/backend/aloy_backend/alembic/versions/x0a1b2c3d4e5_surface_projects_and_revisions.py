"""add Surface projects and immutable source revisions

Revision ID: x0a1b2c3d4e5
Revises: w9a0b1c2d3e4
Create Date: 2026-07-16
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "x0a1b2c3d4e5"
down_revision: Union[str, Sequence[str], None] = "w9a0b1c2d3e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "surface_projects",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("draft_revision_id", sa.String(), nullable=True),
        sa.Column("published_revision_id", sa.String(), nullable=True),
        sa.Column("sdk_version", sa.String(), nullable=False),
        sa.Column("lifecycle", sa.String(), nullable=False),
        sa.Column("user_lock_state", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "organization_id",
            "user_id",
            "event_id",
            name="uq_surface_project_event",
        ),
    )
    for column in (
        "organization_id",
        "user_id",
        "event_id",
        "draft_revision_id",
        "published_revision_id",
        "lifecycle",
        "user_lock_state",
    ):
        op.create_index(
            f"ix_surface_projects_{column}",
            "surface_projects",
            [column],
        )

    op.create_table(
        "surface_revisions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("parent_revision_id", sa.String(), nullable=True),
        sa.Column("creator_run_id", sa.String(), nullable=True),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("request_fingerprint", sa.String(), nullable=False),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("files", sa.JSON(), nullable=False),
        sa.Column("checksum", sa.String(), nullable=False),
        sa.Column("file_count", sa.Integer(), nullable=False),
        sa.Column("total_bytes", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["surface_projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id",
            "idempotency_key",
            name="uq_surface_revision_idempotency",
        ),
        sa.UniqueConstraint(
            "project_id",
            "revision_number",
            name="uq_surface_revision_number",
        ),
    )
    for column in (
        "organization_id",
        "user_id",
        "event_id",
        "project_id",
        "revision_number",
        "parent_revision_id",
        "creator_run_id",
        "checksum",
    ):
        op.create_index(
            f"ix_surface_revisions_{column}",
            "surface_revisions",
            [column],
        )


def downgrade() -> None:
    op.drop_table("surface_revisions")
    op.drop_table("surface_projects")
