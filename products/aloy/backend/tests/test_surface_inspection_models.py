"""Schema coverage for durable trusted Surface inspection evidence."""

import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect

from aloy_backend.models import SurfaceEvidenceArtifact, SurfaceInspection


def test_surface_inspection_models_bind_every_receipt_to_one_exact_bundle():
    inspection = SurfaceInspection(
        organization_id="org_test",
        user_id="user_test",
        event_id="evt_test",
        project_id="surface_test",
        build_id="sbuild_test",
        revision_id="srev_test",
        bundle_key="surface/evt_test/sbuild_test/bundle.zip",
        bundle_sha256="bundle-sha",
        inspection_kind="runtime",
        inspector_version="browser-inspector@1",
        status="passed",
        receipt_sha256="receipt-sha",
    )
    artifact = SurfaceEvidenceArtifact(
        organization_id=inspection.organization_id,
        user_id=inspection.user_id,
        event_id=inspection.event_id,
        project_id=inspection.project_id,
        inspection_id=inspection.id,
        build_id=inspection.build_id,
        revision_id=inspection.revision_id,
        bundle_key=inspection.bundle_key,
        bundle_sha256=inspection.bundle_sha256,
        artifact_kind="viewport_capture",
        storage_key="surface-evidence/sinspect_test/desktop.png",
        content_type="image/png",
        content_sha256="capture-sha",
        size_bytes=1024,
    )

    assert inspection.id.startswith("sinspect_")
    assert artifact.id.startswith("sevidence_")
    assert artifact.bundle_sha256 == inspection.bundle_sha256
    assert artifact.artifact_metadata == {}


def test_surface_inspection_evidence_migration_round_trip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'surface-inspection.db'}")
    metadata = sa.MetaData()
    for table in (
        "events",
        "surface_projects",
        "surface_revisions",
        "surface_builds",
    ):
        sa.Table(table, metadata, sa.Column("id", sa.String(), primary_key=True))
    metadata.create_all(engine)

    with engine.begin() as connection:
        migration = importlib.import_module(
            "aloy_backend.alembic.versions.n6e7f8a9b0c1_surface_inspection_evidence"
        )
        original_op = migration.op
        migration.op = Operations(MigrationContext.configure(connection))
        try:
            migration.upgrade()
            tables = set(inspect(connection).get_table_names())
            assert {"surface_inspections", "surface_evidence_artifacts"} <= tables

            inspection_columns = {
                column["name"]
                for column in inspect(connection).get_columns("surface_inspections")
            }
            assert {
                "organization_id",
                "event_id",
                "project_id",
                "build_id",
                "revision_id",
                "bundle_key",
                "bundle_sha256",
                "receipt_sha256",
            } <= inspection_columns

            evidence_columns = {
                column["name"]
                for column in inspect(connection).get_columns(
                    "surface_evidence_artifacts"
                )
            }
            assert {
                "inspection_id",
                "build_id",
                "revision_id",
                "bundle_key",
                "bundle_sha256",
                "storage_key",
                "content_sha256",
            } <= evidence_columns

            migration.downgrade()
            assert "surface_evidence_artifacts" not in set(
                inspect(connection).get_table_names()
            )
            assert "surface_inspections" not in set(
                inspect(connection).get_table_names()
            )
        finally:
            migration.op = original_op

    engine.dispose()
