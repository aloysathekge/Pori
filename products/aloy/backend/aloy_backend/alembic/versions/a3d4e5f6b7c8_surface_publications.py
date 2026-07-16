"""Add exact Surface publication pointer and append-only release history."""

import sqlalchemy as sa
from alembic import op

revision = "a3d4e5f6b7c8"
down_revision = "z2c3d4e5f6a7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "surface_projects",
        sa.Column("published_build_id", sa.String(), nullable=True),
    )
    op.create_index(
        "ix_surface_projects_published_build_id",
        "surface_projects",
        ["published_build_id"],
    )
    op.create_table(
        "surface_publications",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("revision_id", sa.String(), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("build_id", sa.String(), nullable=False),
        sa.Column("previous_revision_id", sa.String(), nullable=True),
        sa.Column("previous_build_id", sa.String(), nullable=True),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("actor_id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=True),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("request_fingerprint", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["surface_projects.id"]),
        sa.ForeignKeyConstraint(["revision_id"], ["surface_revisions.id"]),
        sa.ForeignKeyConstraint(["build_id"], ["surface_builds.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id",
            "idempotency_key",
            name="uq_surface_publication_idempotency",
        ),
    )
    for column in (
        "organization_id",
        "user_id",
        "event_id",
        "project_id",
        "revision_id",
        "build_id",
        "previous_revision_id",
        "previous_build_id",
        "action",
        "actor_id",
        "run_id",
    ):
        op.create_index(
            f"ix_surface_publications_{column}",
            "surface_publications",
            [column],
        )


def downgrade() -> None:
    op.drop_table("surface_publications")
    op.drop_index(
        "ix_surface_projects_published_build_id",
        table_name="surface_projects",
    )
    op.drop_column("surface_projects", "published_build_id")
