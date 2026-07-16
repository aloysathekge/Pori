from __future__ import annotations

import importlib
import json

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect
from sqlmodel import select

from aloy_backend.models import EventTrailEntry, SurfaceProject, SurfaceRevision
from aloy_backend.run_profiles import SURFACE_BUILDER_RUN_PROFILE
from aloy_backend.runtime import authenticated_run_context
from aloy_backend.surface_authoring import (
    SurfaceAuthoringError,
    SurfaceAuthoringHandler,
    SurfaceConflictError,
    SurfaceWriteFilesParams,
)
from aloy_backend.surface_workspace import resolve_surface_authoring_runtime
from aloy_backend.tools.surface_builds import (
    SURFACE_BUILD_CONTEXT_KEY,
    SURFACE_BUILD_TOOL_NAMES,
)
from aloy_backend.tools.surfaces import (
    SURFACE_AUTHORING_CONTEXT_KEY,
    SURFACE_AUTHORING_TOOL_NAMES,
    register_surface_authoring_tools,
)
from pori import FileErrorCode
from pori.tools.registry import ToolExecutor, ToolRegistry


async def _create_event(client, title: str = "University") -> dict:
    response = await client.post(
        "/v1/events",
        json={
            "title": title,
            "summary": "Manage courses, timetable, and upcoming assessments",
            "phase": "semester",
        },
    )
    assert response.status_code == 201
    return response.json()


def _run_context(event_id: str, *, user_id: str = "test-user"):
    return authenticated_run_context(
        user_id=user_id,
        organization_id=f"user:{user_id}",
        run_id=f"run-surface-{user_id}",
        session_id=f"session-surface-{user_id}",
        event_id=event_id,
        workspace_id=event_id,
        agent_id="surface-builder",
    )


def _write_params(
    *,
    expected_revision: str | None,
    idempotency_key: str,
    content: str,
) -> SurfaceWriteFilesParams:
    return SurfaceWriteFilesParams(
        expected_revision=expected_revision,
        idempotency_key=idempotency_key,
        patches=[
            {
                "path": "/workspace/src/App.tsx",
                "operation": "write",
                "content": content,
            }
        ],
    )


async def test_surface_project_api_is_empty_until_authored_and_tenant_scoped(client):
    event = await _create_event(client)

    empty = await client.get(f"/v1/events/{event['id']}/surface/project")
    denied = await client.get(
        f"/v1/events/{event['id']}/surface/project",
        headers={"X-Test-User": "other-user"},
    )

    assert empty.status_code == 200
    assert empty.json() == {
        "project": None,
        "draft": None,
        "published": None,
        "expected_revision": None,
    }
    assert denied.status_code == 404


async def test_surface_tools_create_immutable_revisions_and_replay_idempotently(
    client,
    db_session_maker,
):
    event = await _create_event(client)
    handler = SurfaceAuthoringHandler(
        run_context=_run_context(event["id"]),
        session_factory=db_session_maker,
    )
    registry = ToolRegistry()
    register_surface_authoring_tools(registry)
    executor = ToolExecutor(registry)
    context = {SURFACE_AUTHORING_CONTEXT_KEY: handler}

    initial = await executor.execute_tool_async(
        "surface_read_project",
        {},
        context,
    )
    assert initial["success"] is True
    assert initial["result"]["expected_revision"] is None

    first_args = {
        "expected_revision": None,
        "idempotency_key": "surface-write-0001",
        "patches": [
            {
                "path": "/workspace/src/App.tsx",
                "content": (
                    "  export default function App() { return <main>Courses</main> }\n"
                ),
            }
        ],
    }
    first = await executor.execute_tool_async(
        "surface_write_files",
        first_args,
        context,
    )
    assert first["success"] is True
    first_revision = first["result"]["draft"]["id"]
    assert first["result"]["changed"] is True
    assert first["result"]["draft"]["revision_number"] == 1

    replay = await executor.execute_tool_async(
        "surface_write_files",
        first_args,
        context,
    )
    assert replay["success"] is True
    assert replay["result"]["replayed"] is True
    assert replay["result"]["draft"]["id"] == first_revision

    second = await handler.write(
        SurfaceWriteFilesParams(
            expected_revision=first_revision,
            idempotency_key="surface-write-0002",
            patches=[
                {
                    "path": "/workspace/src/App.tsx",
                    "content": "export default function App() { return <main>Timetable</main> }",
                },
                {
                    "path": "/workspace/src/styles.css",
                    "content": "main { display: grid; }",
                },
            ],
        )
    )
    second_revision = second["draft"]["id"]
    assert second_revision != first_revision
    assert second["draft"]["parent_revision_id"] == first_revision
    assert second["draft"]["revision_number"] == 2

    public = await client.get(f"/v1/events/{event['id']}/surface/project")
    assert public.status_code == 200
    assert "files" not in public.json()["draft"]
    assert public.json()["draft"]["file_paths"] == [
        "/src/App.tsx",
        "/src/styles.css",
    ]

    async with db_session_maker() as session:
        revisions = list(
            (
                await session.execute(
                    select(SurfaceRevision).order_by(SurfaceRevision.revision_number)
                )
            )
            .scalars()
            .all()
        )
        trail = list(
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.event_id == event["id"],
                        EventTrailEntry.kind == "surface_draft_changed",
                    )
                )
            )
            .scalars()
            .all()
        )
    assert len(revisions) == 2
    assert revisions[0].files["/src/App.tsx"] == (
        "  export default function App() { return <main>Courses</main> }\n"
    )
    assert revisions[1].files["/src/App.tsx"].endswith("Timetable</main> }")
    assert len(trail) == 2
    assert all(entry.run_id == "run-surface-test-user" for entry in trail)


async def test_surface_writes_reject_stale_revisions_and_key_reuse(
    client,
    db_session_maker,
):
    event = await _create_event(client, "Madrid")
    handler = SurfaceAuthoringHandler(
        run_context=_run_context(event["id"]),
        session_factory=db_session_maker,
    )
    first_params = _write_params(
        expected_revision=None,
        idempotency_key="madrid-write-0001",
        content="export default () => <main>Madrid</main>",
    )
    first = await handler.write(first_params)

    with pytest.raises(SurfaceConflictError, match="draft changed"):
        await handler.write(
            _write_params(
                expected_revision=None,
                idempotency_key="madrid-write-stale",
                content="export default () => <main>Stale</main>",
            )
        )
    with pytest.raises(SurfaceConflictError, match="different mutation"):
        await handler.write(
            _write_params(
                expected_revision=first["draft"]["id"],
                idempotency_key="madrid-write-0001",
                content="export default () => <main>Conflicting retry</main>",
            )
        )


async def test_surface_user_lock_blocks_source_mutation(client, db_session_maker):
    event = await _create_event(client, "Career OS")
    handler = SurfaceAuthoringHandler(
        run_context=_run_context(event["id"]),
        session_factory=db_session_maker,
    )
    first = await handler.write(
        _write_params(
            expected_revision=None,
            idempotency_key="career-write-0001",
            content="export default () => <main>Jobs</main>",
        )
    )
    async with db_session_maker() as session:
        project = (await session.execute(select(SurfaceProject))).scalars().one()
        project.user_lock_state = "locked"
        session.add(project)
        await session.commit()

    with pytest.raises(SurfaceAuthoringError, match="locked by the user"):
        await handler.write(
            _write_params(
                expected_revision=first["draft"]["id"],
                idempotency_key="career-write-0002",
                content="export default () => <main>Changed</main>",
            )
        )


async def test_surface_workspace_projects_event_truth_and_current_draft(
    client,
    db_session_maker,
):
    event = await _create_event(client)
    run_context = _run_context(event["id"])
    handler = SurfaceAuthoringHandler(
        run_context=run_context,
        session_factory=db_session_maker,
    )
    await handler.write(
        _write_params(
            expected_revision=None,
            idempotency_key="workspace-write-0001",
            content="export default () => <main>Workspace</main>",
        )
    )

    async with db_session_maker() as session:
        runtime = await resolve_surface_authoring_runtime(
            session,
            run_context=run_context,
            session_factory=db_session_maker,
        )

    event_projection = runtime.file_backend.read("/event/event.json")
    workspace_source = runtime.file_backend.read("/workspace/src/App.tsx")
    denied = runtime.file_backend.write("/event/event.json", "tamper")
    assert event_projection.success
    assert json.loads(event_projection.content or "{}")["title"] == "University"
    assert workspace_source.content == "export default () => <main>Workspace</main>"
    assert denied.error_code is FileErrorCode.PERMISSION_DENIED
    assert (
        runtime.tool_context_extra[SURFACE_AUTHORING_CONTEXT_KEY]
        is runtime.authoring_handler
    )
    assert (
        runtime.tool_context_extra[SURFACE_BUILD_CONTEXT_KEY] is runtime.build_handler
    )

    async with db_session_maker() as session:
        with pytest.raises(ValueError, match="unavailable"):
            await resolve_surface_authoring_runtime(
                session,
                run_context=_run_context(event["id"], user_id="other-user"),
                session_factory=db_session_maker,
            )


def test_surface_source_schema_guards_toolchain_and_workspace_paths():
    with pytest.raises(ValidationError, match="must live under /workspace"):
        SurfaceWriteFilesParams(
            expected_revision=None,
            idempotency_key="invalid-path-0001",
            patches=[
                {
                    "path": "/event/event.json",
                    "content": "tamper",
                }
            ],
        )

    with pytest.raises(ValidationError, match="toolchain file"):
        SurfaceWriteFilesParams(
            expected_revision=None,
            idempotency_key="invalid-toolchain-0001",
            patches=[
                {
                    "path": "/workspace/package.json",
                    "content": "{}",
                }
            ],
        )


def test_surface_builder_profile_explicitly_requires_surface_tools():
    required = SURFACE_AUTHORING_TOOL_NAMES | SURFACE_BUILD_TOOL_NAMES
    assert SURFACE_BUILDER_RUN_PROFILE.required_tools == required
    assert required.issubset(SURFACE_BUILDER_RUN_PROFILE.allowed_tools or frozenset())


def test_surface_migration_creates_and_removes_revision_tables(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'surface-migration.db'}")
    metadata = sa.MetaData()
    sa.Table(
        "events",
        metadata,
        sa.Column("id", sa.String(), primary_key=True),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        migration = importlib.import_module(
            "aloy_backend.alembic.versions."
            "x0a1b2c3d4e5_surface_projects_and_revisions"
        )
        original_op = migration.op
        migration.op = Operations(MigrationContext.configure(connection))
        try:
            migration.upgrade()
            tables = set(inspect(connection).get_table_names())
            assert {"surface_projects", "surface_revisions"} <= tables
            revision_columns = {
                column["name"]
                for column in inspect(connection).get_columns("surface_revisions")
            }
            assert {
                "idempotency_key",
                "request_fingerprint",
                "manifest",
                "files",
                "checksum",
            } <= revision_columns
            migration.downgrade()
            tables = set(inspect(connection).get_table_names())
            assert "surface_projects" not in tables
            assert "surface_revisions" not in tables
        finally:
            migration.op = original_op
    engine.dispose()
