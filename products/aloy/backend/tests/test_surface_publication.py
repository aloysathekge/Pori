from __future__ import annotations

import importlib
import io
import zipfile

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect
from sqlmodel import select

from aloy_backend.models import (
    EventTrailEntry,
    SurfaceProject,
    SurfacePublication,
)
from aloy_backend.runtime import authenticated_run_context
from aloy_backend.surface_authoring import (
    SurfaceAuthoringError,
    SurfaceAuthoringHandler,
    SurfaceConflictError,
    SurfaceWriteFilesParams,
)
from aloy_backend.surface_build_runner import (
    SURFACE_TOOLCHAIN_VERSION,
    SurfaceBuildRunnerResult,
)
from aloy_backend.surface_builds import (
    SurfaceBuildHandler,
    SurfaceBuildParams,
    SurfacePreviewParams,
)
from aloy_backend.surface_publication import SurfacePublicationParams
from aloy_backend.surface_quality import (
    REQUIRED_SURFACE_STATE_VIEWPORTS,
    REQUIRED_SURFACE_VIEWPORTS,
)
from aloy_backend.surface_resource_states import (
    REQUIRED_SURFACE_STATE_FIXTURES,
    SURFACE_STATE_POLICY_VERSION,
)


def _inspection_evidence() -> dict:
    required = [str(item["id"]) for item in REQUIRED_SURFACE_VIEWPORTS]
    return {
        "viewport_matrix": {
            "policy_version": "aloy-surface-viewports@1",
            "required": required,
            "passed": True,
            "viewports": [
                {
                    "id": viewport_id,
                    "capture": {"sha256": f"capture-{viewport_id}"},
                    "accessibility": {
                        "main_landmarks": 1,
                        "unnamed_controls": 0,
                        "images_missing_alt": 0,
                        "keyboard_unreachable": 0,
                        "duplicate_ids": [],
                    },
                    "focus": {
                        "passed": True,
                        "controls": 0,
                        "visited": 0,
                        "visible_indicators": 0,
                    },
                    "contrast": {
                        "passed": True,
                        "failures": 0,
                        "unmeasurable": 0,
                    },
                }
                for viewport_id in required
            ],
        },
        "state_matrix": {
            "policy_version": SURFACE_STATE_POLICY_VERSION,
            "required_states": list(REQUIRED_SURFACE_STATE_FIXTURES),
            "required_viewports": list(REQUIRED_SURFACE_STATE_VIEWPORTS),
            "passed": True,
            "observations": [
                {
                    "state": state,
                    "viewport_id": viewport_id,
                    "fingerprint": f"state-{state}-{viewport_id}",
                    "contrast": {
                        "passed": True,
                        "failures": 0,
                        "unmeasurable": 0,
                    },
                }
                for state in REQUIRED_SURFACE_STATE_FIXTURES
                for viewport_id in REQUIRED_SURFACE_STATE_VIEWPORTS
            ],
        },
        "timings": {
            "policy_version": "aloy-surface-timings@2",
            "runtime_bootstrap_ms": 100.0,
            "viewport_matrix_ms": 200.0,
            "state_matrix_ms": 300.0,
            "interaction_checks_ms": 0.0,
            "primary_jobs_ms": 0.0,
            "total_ms": 600.0,
        },
    }


class FakeBuildRunner:
    toolchain_version = SURFACE_TOOLCHAIN_VERSION

    def __init__(self, bundle: bytes):
        self.bundle = bundle

    async def build(self, *, build_id, files, manifest):
        del build_id, files, manifest
        return SurfaceBuildRunnerResult(
            status="succeeded",
            bundle=self.bundle,
            resource_metrics={
                "runtime_inspection": "passed",
                "interaction_inspection": "passed",
                "viewport_inspection": "passed",
                "accessibility_inspection": "passed",
                "state_inspection": "passed",
                "focus_inspection": "passed",
                "contrast_inspection": "passed",
                "inspection_evidence": _inspection_evidence(),
            },
        )


class MemoryObjectStore:
    def __init__(self):
        self.values: dict[str, bytes] = {}

    def put(self, key, data, *, content_type):
        del content_type
        value = data.read()
        self.values[key] = value
        return len(value)

    def open(self, key):
        return io.BytesIO(self.values[key])

    def delete(self, key):
        self.values.pop(key, None)

    def url(self, key, *, expires_s=300):
        del key, expires_s
        return None


def _bundle(label: str) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "surface.js",
            f'document.getElementById("root").textContent = "{label}";',
        )
        archive.writestr("surface.css", "body { margin: 0; }")
    return output.getvalue()


async def _create_event(client) -> dict:
    response = await client.post(
        "/v1/events",
        json={"title": "University", "summary": "Persistent Event"},
    )
    assert response.status_code == 201
    return response.json()


def _run_context(event_id: str):
    return authenticated_run_context(
        user_id="test-user",
        organization_id="user:test-user",
        run_id="run-surface-publication",
        session_id="session-surface-publication",
        event_id=event_id,
        workspace_id=event_id,
        agent_id="surface-builder",
    )


async def _write_revision(
    author: SurfaceAuthoringHandler,
    *,
    expected_revision: str | None,
    key: str,
    label: str,
) -> str:
    result = await author.write(
        SurfaceWriteFilesParams(
            expected_revision=expected_revision,
            idempotency_key=key,
            patches=[
                {
                    "path": "/workspace/src/App.tsx",
                    "content": f"export default () => <main>{label}</main>",
                }
            ],
        )
    )
    return result["draft"]["id"]


async def test_publication_keeps_drafts_off_live_runtime_and_rolls_back_last_good(
    client,
    db_session_maker,
    monkeypatch,
):
    event = await _create_event(client)
    store = MemoryObjectStore()
    author = SurfaceAuthoringHandler(
        run_context=_run_context(event["id"]),
        session_factory=db_session_maker,
    )
    first_revision = await _write_revision(
        author,
        expected_revision=None,
        key="publish-author-0001",
        label="Timetable",
    )
    first_bundle = _bundle("Timetable")
    builder = SurfaceBuildHandler(
        run_context=_run_context(event["id"]),
        runner=FakeBuildRunner(first_bundle),
        object_store=store,
        session_factory=db_session_maker,
    )
    first_build = await builder.build(
        SurfaceBuildParams(
            revision_id=first_revision,
            idempotency_key="publish-build-0001",
        )
    )
    unpublished_context = await client.get(
        f"/v1/events/{event['id']}/surface/context",
        params={"build_id": first_build["id"]},
    )
    assert unpublished_context.status_code == 404
    preview = await builder.preview(SurfacePreviewParams(build_id=first_build["id"]))
    assert preview["quality_gate"]["passed"] is True
    published = await builder.publish(
        SurfacePublicationParams(
            build_id=first_build["id"],
            expected_published_revision_id=None,
            expected_published_build_id=None,
            idempotency_key="publish-release-0001",
        )
    )
    replay = await builder.publish(
        SurfacePublicationParams(
            build_id=first_build["id"],
            expected_published_revision_id=None,
            expected_published_build_id=None,
            idempotency_key="publish-release-0001",
        )
    )
    assert published["action"] == "publish"
    assert replay["replayed"] is True

    live = await client.get(f"/v1/events/{event['id']}/surface/runtime")
    assert live.status_code == 200
    assert live.json()["build"]["id"] == first_build["id"]
    published_context = await client.get(
        f"/v1/events/{event['id']}/surface/context",
        params={"build_id": first_build["id"]},
    )
    assert published_context.status_code == 200

    second_revision = await _write_revision(
        author,
        expected_revision=first_revision,
        key="publish-author-0002",
        label="Exams",
    )
    builder._runner = FakeBuildRunner(_bundle("Exams"))
    second_build = await builder.build(
        SurfaceBuildParams(
            revision_id=second_revision,
            idempotency_key="publish-build-0002",
        )
    )
    second_preview = await builder.preview(
        SurfacePreviewParams(build_id=second_build["id"])
    )
    assert second_preview["quality_gate"]["passed"] is True

    still_live = await client.get(f"/v1/events/{event['id']}/surface/runtime")
    assert still_live.json()["build"]["id"] == first_build["id"]

    with pytest.raises(SurfaceConflictError, match="Published Surface changed"):
        await builder.publish(
            SurfacePublicationParams(
                build_id=second_build["id"],
                expected_published_revision_id=None,
                expected_published_build_id=None,
                idempotency_key="publish-release-stale-0002",
            )
        )
    second_publish = await builder.publish(
        SurfacePublicationParams(
            build_id=second_build["id"],
            expected_published_revision_id=first_revision,
            expected_published_build_id=first_build["id"],
            idempotency_key="publish-release-0002",
        )
    )
    assert second_publish["previous_build_id"] == first_build["id"]

    monkeypatch.setattr("aloy_backend.routes.surfaces.get_object_store", lambda: store)
    rollback_body = {
        "build_id": first_build["id"],
        "expected_published_revision_id": second_revision,
        "expected_published_build_id": second_build["id"],
        "idempotency_key": "publish-rollback-0001",
    }
    restored = await client.post(
        f"/v1/events/{event['id']}/surface/rollback",
        json=rollback_body,
    )
    restored_replay = await client.post(
        f"/v1/events/{event['id']}/surface/rollback",
        json=rollback_body,
    )
    assert restored.status_code == 200, restored.text
    assert restored.json()["action"] == "rollback"
    assert restored_replay.json()["replayed"] is True

    runtime = await client.get(f"/v1/events/{event['id']}/surface/runtime")
    history = await client.get(f"/v1/events/{event['id']}/surface/publications")
    assert runtime.json()["build"]["id"] == first_build["id"]
    assert [item["action"] for item in history.json()] == [
        "rollback",
        "publish",
        "publish",
    ]

    async with db_session_maker() as session:
        project = (await session.execute(select(SurfaceProject))).scalars().one()
        assert project.published_revision_id == first_revision
        assert project.published_build_id == first_build["id"]
        publications = (
            (await session.execute(select(SurfacePublication))).scalars().all()
        )
        assert len(publications) == 3
        trail = list(
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.event_id == event["id"],
                        EventTrailEntry.kind.in_(
                            ["surface_published", "surface_rolled_back"]
                        ),
                    )
                )
            )
            .scalars()
            .all()
        )
        assert {entry.kind for entry in trail} == {
            "surface_published",
            "surface_rolled_back",
        }


async def test_publication_rejects_missing_or_tampered_artifact_without_moving_pointer(
    client,
    db_session_maker,
):
    event = await _create_event(client)
    store = MemoryObjectStore()
    author = SurfaceAuthoringHandler(
        run_context=_run_context(event["id"]),
        session_factory=db_session_maker,
    )
    revision = await _write_revision(
        author,
        expected_revision=None,
        key="invalid-author-0001",
        label="Courses",
    )
    builder = SurfaceBuildHandler(
        run_context=_run_context(event["id"]),
        runner=FakeBuildRunner(_bundle("Courses")),
        object_store=store,
        session_factory=db_session_maker,
    )
    build = await builder.build(
        SurfaceBuildParams(
            revision_id=revision,
            idempotency_key="invalid-build-0001",
        )
    )
    preview = await builder.preview(SurfacePreviewParams(build_id=build["id"]))
    assert preview["quality_gate"]["passed"] is True
    async with db_session_maker() as session:
        project = (await session.execute(select(SurfaceProject))).scalars().one()
        assert project.published_build_id is None
    store.values[next(iter(store.values))] = b"tampered"

    try:
        await builder.publish(
            SurfacePublicationParams(
                build_id=build["id"],
                idempotency_key="invalid-release-0001",
            )
        )
        raise AssertionError("tampered publication unexpectedly succeeded")
    except SurfaceAuthoringError as exc:
        assert "checksum failed" in str(exc)

    async with db_session_maker() as session:
        project = (await session.execute(select(SurfaceProject))).scalars().one()
        assert project.published_build_id is None
        assert (await session.execute(select(SurfacePublication))).scalars().all() == []


async def test_publication_rejects_build_without_exact_quality_receipt(
    client,
    db_session_maker,
):
    event = await _create_event(client)
    store = MemoryObjectStore()
    author = SurfaceAuthoringHandler(
        run_context=_run_context(event["id"]),
        session_factory=db_session_maker,
    )
    revision = await _write_revision(
        author,
        expected_revision=None,
        key="quality-author-0001",
        label="Quality gate",
    )
    builder = SurfaceBuildHandler(
        run_context=_run_context(event["id"]),
        runner=FakeBuildRunner(_bundle("Quality gate")),
        object_store=store,
        session_factory=db_session_maker,
    )
    build = await builder.build(
        SurfaceBuildParams(
            revision_id=revision,
            idempotency_key="quality-build-0001",
        )
    )

    with pytest.raises(SurfaceAuthoringError, match="no trusted quality receipt"):
        await builder.publish(
            SurfacePublicationParams(
                build_id=build["id"],
                expected_published_revision_id=None,
                expected_published_build_id=None,
                idempotency_key="quality-release-0001",
            )
        )

    preview = await builder.preview(SurfacePreviewParams(build_id=build["id"]))
    assert preview["quality_gate"]["binding"]["build_id"] == build["id"]
    published = await builder.publish(
        SurfacePublicationParams(
            build_id=build["id"],
            expected_published_revision_id=None,
            expected_published_build_id=None,
            idempotency_key="quality-release-0002",
        )
    )
    assert published["build_id"] == build["id"]


def test_surface_publication_migration_round_trip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'surface-publication.db'}")
    metadata = sa.MetaData()
    for table in ("events", "surface_projects", "surface_revisions", "surface_builds"):
        sa.Table(table, metadata, sa.Column("id", sa.String(), primary_key=True))
    metadata.create_all(engine)

    with engine.begin() as connection:
        migration = importlib.import_module(
            "aloy_backend.alembic.versions.a3d4e5f6b7c8_surface_publications"
        )
        original_op = migration.op
        migration.op = Operations(MigrationContext.configure(connection))
        try:
            migration.upgrade()
            assert "surface_publications" in inspect(connection).get_table_names()
            project_columns = {
                column["name"]
                for column in inspect(connection).get_columns("surface_projects")
            }
            assert "published_build_id" in project_columns
            publication_columns = {
                column["name"]
                for column in inspect(connection).get_columns("surface_publications")
            }
            assert {
                "build_id",
                "revision_id",
                "revision_number",
                "previous_build_id",
                "idempotency_key",
                "request_fingerprint",
            } <= publication_columns
            migration.downgrade()
            assert "surface_publications" not in inspect(connection).get_table_names()
        finally:
            migration.op = original_op
    engine.dispose()
