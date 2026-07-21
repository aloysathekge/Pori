"""Persist durable Surface evolution proposals.

Revision ID: o7f8a9b0c1d2
Revises: n6e7f8a9b0c1
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "o7f8a9b0c1d2"
down_revision: str | None = "n6e7f8a9b0c1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "surface_evolution_proposals",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("trigger", sa.String(), nullable=False),
        sa.Column("goal", sa.String(), nullable=False),
        sa.Column("signal_fingerprint", sa.String(), nullable=False),
        sa.Column("decision_fingerprint", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("occurrence_count", sa.Integer(), nullable=False),
        sa.Column("base_revision_id", sa.String(), nullable=True),
        sa.Column("base_build_id", sa.String(), nullable=True),
        sa.Column("base_data_revision", sa.Integer(), nullable=False),
        sa.Column("reason_codes", sa.JSON(), nullable=False),
        sa.Column("evidence_refs", sa.JSON(), nullable=False),
        sa.Column("builder_run_id", sa.String(), nullable=True),
        sa.Column("decided_by", sa.String(), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cooldown_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["surface_projects.id"]),
        sa.ForeignKeyConstraint(["base_revision_id"], ["surface_revisions.id"]),
        sa.ForeignKeyConstraint(["base_build_id"], ["surface_builds.id"]),
        sa.ForeignKeyConstraint(["builder_run_id"], ["runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "event_id",
            "signal_fingerprint",
            name="uq_surface_evolution_signal",
        ),
    )
    for column in (
        "organization_id",
        "user_id",
        "event_id",
        "project_id",
        "trigger",
        "signal_fingerprint",
        "decision_fingerprint",
        "status",
        "base_revision_id",
        "base_build_id",
        "builder_run_id",
        "decided_by",
    ):
        op.create_index(
            f"ix_surface_evolution_proposals_{column}",
            "surface_evolution_proposals",
            [column],
        )


def downgrade() -> None:
    op.drop_table("surface_evolution_proposals")
