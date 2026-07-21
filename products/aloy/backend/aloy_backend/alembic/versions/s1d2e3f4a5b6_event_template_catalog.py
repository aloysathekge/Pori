"""Add the versioned Event template catalog and installation receipts.

Revision ID: s1d2e3f4a5b6
Revises: r0c1d2e3f4a5
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "s1d2e3f4a5b6"
down_revision: str | None = "r0c1d2e3f4a5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> tuple[sa.Column, ...]:
    return (sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),)


def upgrade() -> None:
    op.create_table(
        "event_templates",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("slug", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("summary", sa.String(), nullable=False),
        sa.Column("discovery_group", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("current_release_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug"),
    )
    op.create_index("ix_event_templates_slug", "event_templates", ["slug"], unique=True)
    op.create_index(
        "ix_event_templates_discovery_group",
        "event_templates",
        ["discovery_group"],
    )
    op.create_index("ix_event_templates_status", "event_templates", ["status"])
    op.create_index(
        "ix_event_templates_current_release_id",
        "event_templates",
        ["current_release_id"],
    )

    op.create_table(
        "event_template_releases",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("template_id", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("release_notes", sa.String(), nullable=False),
        sa.Column("checksum", sa.String(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["template_id"], ["event_templates.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "template_id", "version", name="uq_event_template_release_version"
        ),
    )
    op.create_index(
        "ix_event_template_releases_template_id",
        "event_template_releases",
        ["template_id"],
    )
    op.create_index(
        "ix_event_template_releases_version",
        "event_template_releases",
        ["version"],
    )
    op.create_index(
        "ix_event_template_releases_status",
        "event_template_releases",
        ["status"],
    )
    op.create_index(
        "ix_event_template_releases_checksum",
        "event_template_releases",
        ["checksum"],
    )

    op.create_table(
        "event_template_assets",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("release_id", sa.String(), nullable=False),
        sa.Column("asset_key", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("storage_key", sa.String(), nullable=False),
        sa.Column("content_type", sa.String(), nullable=False),
        sa.Column("sha256", sa.String(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["release_id"], ["event_template_releases.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "release_id", "asset_key", name="uq_event_template_release_asset"
        ),
    )
    op.create_index(
        "ix_event_template_assets_release_id",
        "event_template_assets",
        ["release_id"],
    )
    op.create_index(
        "ix_event_template_assets_asset_key",
        "event_template_assets",
        ["asset_key"],
    )
    op.create_index("ix_event_template_assets_kind", "event_template_assets", ["kind"])
    op.create_index(
        "ix_event_template_assets_sha256", "event_template_assets", ["sha256"]
    )

    op.create_table(
        "event_template_compatibility",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("release_id", sa.String(), nullable=False),
        sa.Column("requirement_key", sa.String(), nullable=False),
        sa.Column("requirement", sa.JSON(), nullable=False),
        sa.Column("required", sa.Boolean(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["release_id"], ["event_template_releases.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "release_id",
            "requirement_key",
            name="uq_event_template_compatibility",
        ),
    )
    op.create_index(
        "ix_event_template_compatibility_release_id",
        "event_template_compatibility",
        ["release_id"],
    )
    op.create_index(
        "ix_event_template_compatibility_requirement_key",
        "event_template_compatibility",
        ["requirement_key"],
    )

    op.create_table(
        "event_template_seeds",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("release_id", sa.String(), nullable=False),
        sa.Column("seed_key", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["release_id"], ["event_template_releases.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "release_id", "seed_key", name="uq_event_template_release_seed"
        ),
    )
    op.create_index(
        "ix_event_template_seeds_release_id",
        "event_template_seeds",
        ["release_id"],
    )
    op.create_index(
        "ix_event_template_seeds_seed_key",
        "event_template_seeds",
        ["seed_key"],
    )
    op.create_index("ix_event_template_seeds_kind", "event_template_seeds", ["kind"])
    op.create_index(
        "ix_event_template_seeds_ordinal", "event_template_seeds", ["ordinal"]
    )

    op.create_table(
        "event_template_guided_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("release_id", sa.String(), nullable=False),
        sa.Column("job_key", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("instructions", sa.String(), nullable=False),
        sa.Column("definition_of_done", sa.String(), nullable=False),
        sa.Column("priority", sa.String(), nullable=False),
        sa.Column("execution_profile", sa.String(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("materialize_task", sa.Boolean(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["release_id"], ["event_template_releases.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "release_id",
            "job_key",
            name="uq_event_template_release_guided_job",
        ),
    )
    op.create_index(
        "ix_event_template_guided_jobs_release_id",
        "event_template_guided_jobs",
        ["release_id"],
    )
    op.create_index(
        "ix_event_template_guided_jobs_job_key",
        "event_template_guided_jobs",
        ["job_key"],
    )
    op.create_index(
        "ix_event_template_guided_jobs_priority",
        "event_template_guided_jobs",
        ["priority"],
    )
    op.create_index(
        "ix_event_template_guided_jobs_execution_profile",
        "event_template_guided_jobs",
        ["execution_profile"],
    )
    op.create_index(
        "ix_event_template_guided_jobs_ordinal",
        "event_template_guided_jobs",
        ["ordinal"],
    )

    op.create_table(
        "event_template_installations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("template_id", sa.String(), nullable=False),
        sa.Column("release_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("release_snapshot", sa.JSON(), nullable=False),
        sa.Column("installed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "organization_id",
            "user_id",
            "idempotency_key",
            name="uq_event_template_installation_request",
        ),
        sa.UniqueConstraint("event_id", name="uq_event_template_installation_event"),
    )
    for column in (
        "organization_id",
        "user_id",
        "template_id",
        "release_id",
        "event_id",
        "status",
    ):
        op.create_index(
            f"ix_event_template_installations_{column}",
            "event_template_installations",
            [column],
        )


def downgrade() -> None:
    op.drop_table("event_template_installations")
    op.drop_table("event_template_guided_jobs")
    op.drop_table("event_template_seeds")
    op.drop_table("event_template_compatibility")
    op.drop_table("event_template_assets")
    op.drop_table("event_template_releases")
    op.drop_table("event_templates")
