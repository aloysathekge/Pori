"""add enterprise tenancy, policies, and worker leases

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-06-20
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, Sequence[str], None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("slug", sa.String(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("policy", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_organizations_slug", "organizations", ["slug"], unique=True)
    op.create_index("ix_organizations_created_by", "organizations", ["created_by"])
    op.create_table(
        "organization_memberships",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("organization_id", "user_id", name="uq_org_membership"),
    )
    for column in ("organization_id", "user_id", "role", "status"):
        op.create_index(
            f"ix_organization_memberships_{column}",
            "organization_memberships",
            [column],
        )

    scoped_tables = (
        "conversations",
        "agent_configs",
        "team_configs",
        "usage_records",
        "core_memory_blocks",
    )
    for table in scoped_tables:
        op.add_column(table, sa.Column("organization_id", sa.String(), nullable=True))

    op.add_column("runs", sa.Column("team_config_id", sa.String(), nullable=True))
    op.add_column(
        "runs",
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "runs",
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
    )
    op.add_column(
        "runs",
        sa.Column(
            "timeout_seconds", sa.Integer(), nullable=False, server_default="900"
        ),
    )
    op.add_column("runs", sa.Column("lease_owner", sa.String(), nullable=True))
    op.add_column(
        "runs", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "runs", sa.Column("started_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "runs", sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "runs",
        sa.Column(
            "cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
    )
    op.add_column(
        "runs",
        sa.Column(
            "isolation_profile",
            sa.String(),
            nullable=False,
            server_default="worker-process",
        ),
    )

    connection = op.get_bind()
    user_ids: set[str] = set()
    for table in (
        "user_profiles",
        "conversations",
        "agent_configs",
        "team_configs",
        "usage_records",
        "core_memory_blocks",
        "runs",
        "trace_records",
        "knowledge_entries",
    ):
        column = "id" if table == "user_profiles" else "user_id"
        rows = connection.execute(sa.text(f"SELECT DISTINCT {column} FROM {table}"))
        user_ids.update(str(row[0]) for row in rows if row[0])

    now = datetime.now(timezone.utc)
    organizations = sa.table(
        "organizations",
        sa.column("id"),
        sa.column("name"),
        sa.column("slug"),
        sa.column("created_by"),
        sa.column("policy"),
        sa.column("created_at"),
        sa.column("updated_at"),
    )
    memberships = sa.table(
        "organization_memberships",
        sa.column("id"),
        sa.column("organization_id"),
        sa.column("user_id"),
        sa.column("role"),
        sa.column("status"),
        sa.column("created_at"),
        sa.column("updated_at"),
    )
    for user_id in sorted(user_ids):
        digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
        organization_id = f"user:{user_id}"
        connection.execute(
            organizations.insert().values(
                id=organization_id,
                name="Personal Workspace",
                slug=f"personal-{digest[:20]}",
                created_by=user_id,
                policy={},
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            memberships.insert().values(
                id=digest[:32],
                organization_id=organization_id,
                user_id=user_id,
                role="owner",
                status="active",
                created_at=now,
                updated_at=now,
            )
        )

    for table in scoped_tables:
        op.execute(f"UPDATE {table} SET organization_id = 'user:' || user_id")
        with op.batch_alter_table(table) as batch_op:
            batch_op.alter_column(
                "organization_id", existing_type=sa.String(), nullable=False
            )
        op.create_index(f"ix_{table}_organization_id", table, ["organization_id"])

    for column in ("team_config_id", "lease_owner"):
        op.create_index(f"ix_runs_{column}", "runs", [column])


def downgrade() -> None:
    for column in ("lease_owner", "team_config_id"):
        op.drop_index(f"ix_runs_{column}", table_name="runs")
    with op.batch_alter_table("runs") as batch_op:
        for column in (
            "isolation_profile",
            "cancel_requested",
            "completed_at",
            "started_at",
            "lease_expires_at",
            "lease_owner",
            "max_attempts",
            "timeout_seconds",
            "attempt_count",
            "team_config_id",
        ):
            batch_op.drop_column(column)
    for table in (
        "core_memory_blocks",
        "usage_records",
        "team_configs",
        "agent_configs",
        "conversations",
    ):
        op.drop_index(f"ix_{table}_organization_id", table_name=table)
        with op.batch_alter_table(table) as batch_op:
            batch_op.drop_column("organization_id")
    for column in ("status", "role", "user_id", "organization_id"):
        op.drop_index(
            f"ix_organization_memberships_{column}",
            table_name="organization_memberships",
        )
    op.drop_table("organization_memberships")
    op.drop_index("ix_organizations_created_by", table_name="organizations")
    op.drop_index("ix_organizations_slug", table_name="organizations")
    op.drop_table("organizations")
