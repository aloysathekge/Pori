"""add governed evolution

Revision ID: d0e1f2a3b4c5
Revises: c9d0e1f2a3b4
Create Date: 2026-06-23
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d0e1f2a3b4c5"
down_revision: Union[str, Sequence[str], None] = "c9d0e1f2a3b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "evolution_proposals",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("artifact_kind", sa.String(), nullable=False),
        sa.Column("target", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("summary", sa.String(), nullable=False),
        sa.Column("rationale", sa.String(), nullable=False),
        sa.Column("current_version", sa.String(), nullable=True),
        sa.Column("proposed_version", sa.String(), nullable=False),
        sa.Column("proposed_content", sa.String(), nullable=False),
        sa.Column("eval_cases", sa.JSON(), nullable=False),
        sa.Column("eval_results", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("approved_by", sa.String(), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("supersedes_proposal_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "organization_id",
            "target",
            "proposed_version",
            name="uq_evolution_proposal_version",
        ),
    )
    for column in (
        "organization_id",
        "created_by",
        "artifact_kind",
        "target",
        "proposed_version",
        "status",
        "approved_by",
        "supersedes_proposal_id",
    ):
        op.create_index(
            f"ix_evolution_proposals_{column}",
            "evolution_proposals",
            [column],
        )

    op.create_table(
        "evolution_activations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("target", sa.String(), nullable=False),
        sa.Column("proposal_id", sa.String(), nullable=False),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("activated_by", sa.String(), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("rolled_back_at", sa.DateTime(timezone=True), nullable=True),
    )
    for column in ("organization_id", "target", "proposal_id", "activated_by"):
        op.create_index(
            f"ix_evolution_activations_{column}",
            "evolution_activations",
            [column],
        )


def downgrade() -> None:
    op.drop_table("evolution_activations")
    op.drop_table("evolution_proposals")
