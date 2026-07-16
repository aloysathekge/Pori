"""add Surface SDK data and interaction ledger

Revision ID: z2c3d4e5f6a7
Revises: y1b2c3d4e5f6
Create Date: 2026-07-16
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "z2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "y1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "surface_projects",
        sa.Column("data_revision", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_table(
        "surface_data_records",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("namespace", sa.String(), nullable=False),
        sa.Column("record_key", sa.String(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("posture", sa.String(), nullable=False),
        sa.Column("actor_id", sa.String(), nullable=False),
        sa.Column("provenance", sa.JSON(), nullable=False),
        sa.Column("evidence_refs", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["surface_projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id", "namespace", "record_key", name="uq_surface_data_record"
        ),
    )
    for column in (
        "organization_id",
        "user_id",
        "event_id",
        "project_id",
        "namespace",
        "record_key",
        "revision",
        "posture",
        "actor_id",
    ):
        op.create_index(
            f"ix_surface_data_records_{column}", "surface_data_records", [column]
        )

    op.create_table(
        "surface_interactions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("build_id", sa.String(), nullable=False),
        sa.Column("code_revision_id", sa.String(), nullable=False),
        sa.Column("conversation_id", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("interaction_class", sa.String(), nullable=False),
        sa.Column("component_id", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("actor_id", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("request_fingerprint", sa.String(), nullable=False),
        sa.Column("base_data_revision", sa.Integer(), nullable=False),
        sa.Column("result_data_revision", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("handling_run_id", sa.String(), nullable=True),
        sa.Column("proposal_id", sa.String(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["build_id"], ["surface_builds.id"]),
        sa.ForeignKeyConstraint(["code_revision_id"], ["surface_revisions.id"]),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["surface_projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id",
            "idempotency_key",
            name="uq_surface_interaction_idempotency",
        ),
    )
    for column in (
        "organization_id",
        "user_id",
        "event_id",
        "project_id",
        "build_id",
        "code_revision_id",
        "conversation_id",
        "name",
        "interaction_class",
        "actor_id",
        "result_data_revision",
        "status",
        "handling_run_id",
        "proposal_id",
    ):
        op.create_index(
            f"ix_surface_interactions_{column}", "surface_interactions", [column]
        )


def downgrade() -> None:
    op.drop_table("surface_interactions")
    op.drop_table("surface_data_records")
    op.drop_column("surface_projects", "data_revision")
