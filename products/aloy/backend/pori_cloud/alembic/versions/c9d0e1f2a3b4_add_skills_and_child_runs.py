"""add skills and child runs

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
Create Date: 2026-06-23
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c9d0e1f2a3b4"
down_revision: Union[str, Sequence[str], None] = "b8c9d0e1f2a3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "skill_definitions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("slug", sa.String(), nullable=False),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("summary", sa.String(), nullable=False),
        sa.Column("instructions", sa.String(), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("required_tools", sa.JSON(), nullable=False),
        sa.Column("required_credentials", sa.JSON(), nullable=False),
        sa.Column("required_platforms", sa.JSON(), nullable=False),
        sa.Column("required_model_capabilities", sa.JSON(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("sensitivity", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "organization_id", "slug", "version", name="uq_skill_version"
        ),
    )
    for column in ("organization_id", "created_by", "slug", "version", "status"):
        op.create_index(f"ix_skill_definitions_{column}", "skill_definitions", [column])
    op.create_table(
        "skill_grants",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("skill_id", sa.String(), nullable=False),
        sa.Column("principal_type", sa.String(), nullable=False),
        sa.Column("principal_id", sa.String(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "organization_id",
            "skill_id",
            "principal_type",
            "principal_id",
            name="uq_skill_grant",
        ),
    )
    for column in (
        "organization_id",
        "skill_id",
        "principal_type",
        "principal_id",
        "created_by",
    ):
        op.create_index(f"ix_skill_grants_{column}", "skill_grants", [column])
    with op.batch_alter_table("runs") as batch:
        batch.add_column(sa.Column("parent_run_id", sa.String(), nullable=True))
        batch.add_column(sa.Column("root_run_id", sa.String(), nullable=True))
        batch.add_column(sa.Column("idempotency_key", sa.String(), nullable=True))
        batch.add_column(
            sa.Column("child_depth", sa.Integer(), nullable=False, server_default="0")
        )
        for column in ("parent_run_id", "root_run_id", "idempotency_key"):
            batch.create_index(f"ix_runs_{column}", [column])
    with op.batch_alter_table("runs") as batch:
        batch.alter_column("child_depth", server_default=None)


def downgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        for column in ("idempotency_key", "root_run_id", "parent_run_id"):
            batch.drop_index(f"ix_runs_{column}")
        batch.drop_column("child_depth")
        batch.drop_column("idempotency_key")
        batch.drop_column("root_run_id")
        batch.drop_column("parent_run_id")
    op.drop_table("skill_grants")
    op.drop_table("skill_definitions")
