from __future__ import annotations

import hashlib
import importlib
import io
import shutil
import zipfile

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect
from sqlmodel import select

from aloy_backend.models import EventTrailEntry, SurfaceBuild
from aloy_backend.runtime import authenticated_run_context
from aloy_backend.surface_authoring import (
    SurfaceAuthoringHandler,
    SurfaceConflictError,
    SurfaceWriteFilesParams,
)
from aloy_backend.surface_build_runner import (
    MAX_SURFACE_BUNDLE_BYTES,
    SURFACE_TOOLCHAIN_VERSION,
    LocalDevelopmentSurfaceBuildRunner,
    SandboxSurfaceBuildRunner,
    SurfaceBuildRunnerResult,
    UnavailableSurfaceBuildRunner,
    configured_surface_build_runner,
    validate_surface_source,
)
from aloy_backend.surface_builds import (
    SurfaceBuildHandler,
    SurfaceBuildParams,
    SurfacePreviewParams,
)
from aloy_backend.surface_manifest import SurfaceManifest
from aloy_backend.surface_runtime import (
    InvalidSurfaceBundle,
    build_surface_runtime_document,
)
from aloy_backend.surface_runtime_inspection import inspect_surface_runtime
from pori import LocalSandboxProvider


class FakeBuildRunner:
    toolchain_version = SURFACE_TOOLCHAIN_VERSION

    def __init__(self, result: SurfaceBuildRunnerResult):
        self.result = result
        self.calls: list[dict] = []

    async def build(self, *, build_id, files, manifest):
        self.calls.append({"build_id": build_id, "files": files, "manifest": manifest})
        return self.result


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


async def test_local_development_builder_fails_closed_without_pinned_toolchain(
    tmp_path,
):
    result = await LocalDevelopmentSurfaceBuildRunner(repository_root=tmp_path).build(
        build_id="missing-local-toolchain",
        files={"/src/App.tsx": "export default () => <main />"},
        manifest={},
    )

    assert result.status == "blocked"
    assert result.diagnostics[0]["code"] == "local_toolchain_unavailable"


async def test_local_development_bundle_is_browser_safe():
    if shutil.which("node") is None:
        pytest.skip("Node.js is not installed")
    result = await LocalDevelopmentSurfaceBuildRunner().build(
        build_id="browser-safe-local-toolchain",
        files={
            "/src/App.tsx": (
                'import React from "react"; '
                "export default () => <main>Aloy Surface</main>"
            )
        },
        manifest={},
    )
    if result.status == "blocked":
        pytest.skip("Pinned Aloy app dependencies are not installed")

    assert result.status == "succeeded"
    assert result.bundle is not None
    with zipfile.ZipFile(io.BytesIO(result.bundle)) as archive:
        script = archive.read("surface.js").decode("utf-8")
    assert "process.env.NODE_ENV" not in script
    assert "process is not defined" not in script
    assert "aloy.surface.connect" in script
    runtime = build_surface_runtime_document(result.bundle)
    diagnostics = inspect_surface_runtime(
        runtime,
        {
            "protocol_version": "1",
            "sdk_version": "1",
            "event_id": "event-smoke",
            "project_id": "project-smoke",
            "build_id": "build-smoke",
            "code_revision_id": "revision-smoke",
            "data_revision": 0,
            "capabilities": [],
            "widgets": [],
            "data": {"interactions": []},
        },
    )
    assert diagnostics == []


async def test_local_browser_gate_rejects_render_exception():
    if shutil.which("node") is None:
        pytest.skip("Node.js is not installed")
    result = await LocalDevelopmentSurfaceBuildRunner().build(
        build_id="runtime-exception-local-toolchain",
        files={
            "/src/App.tsx": (
                "export default function App() { " "throw new Error('render failed'); }"
            )
        },
        manifest={},
    )
    if result.status == "blocked":
        pytest.skip("Pinned Aloy app dependencies are not installed")
    assert result.bundle is not None

    diagnostics = inspect_surface_runtime(
        build_surface_runtime_document(result.bundle),
        {
            "protocol_version": "1",
            "sdk_version": "1",
            "event_id": "event-smoke",
            "project_id": "project-smoke",
            "build_id": "build-smoke",
            "code_revision_id": "revision-smoke",
            "data_revision": 0,
            "capabilities": [],
            "widgets": [],
            "data": {"interactions": []},
        },
    )
    assert {item["code"] for item in diagnostics} == {"runtime_exception"}
    assert "render failed" in diagnostics[0]["message"]


async def test_local_browser_gate_executes_accessible_interaction_checks():
    if shutil.which("node") is None:
        pytest.skip("Node.js is not installed")
    manifest = SurfaceManifest.model_validate(
        {
            "capabilities": ["data:career"],
            "intents": {
                "career.application_created": {
                    "class": "state",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "applicationId": {"type": "string"},
                            "company": {"type": "string"},
                        },
                        "required": ["applicationId", "company"],
                        "additionalProperties": False,
                    },
                    "write": {
                        "namespace": "career",
                        "operation": "create",
                        "key_field": "applicationId",
                    },
                }
            },
            "interaction_checks": [
                {
                    "name": "Add an application",
                    "steps": [
                        {
                            "action": "fill",
                            "role": "textbox",
                            "name": "Company",
                            "value": "Aloy Verification",
                        },
                        {"action": "click", "role": "button", "name": "Save"},
                    ],
                    "expect": {
                        "method": "command",
                        "name": "career.application_created",
                    },
                }
            ],
        }
    )
    files = {
        "/src/App.tsx": (
            'import React, { useState } from "react"; '
            'import { useSurfaceCommand } from "@aloy/surface"; '
            "export default function App(){const [company,setCompany]=useState('');"
            "const save=useSurfaceCommand('career.application_created',{componentId:'add-application'});"
            "return <form onSubmit={async event=>{event.preventDefault();try{await save.execute("
            "{applicationId:'smoke',company});}catch{}}}>"
            "<label>Company<input aria-label='Company' value={company} "
            "onChange={event=>setCompany(event.target.value)}/></label>"
            "<button type='submit' disabled={save.pending}>Save</button>"
            "<p {...save.feedbackProps}>{save.status==='pending'?'Saving':"
            "save.status==='committed'?'Application saved':save.error?.message||'Ready'}</p>"
            "</form>}"
        )
    }
    result = await LocalDevelopmentSurfaceBuildRunner().build(
        build_id="interactive-local-toolchain",
        files=files,
        manifest=manifest.model_dump(mode="json", by_alias=True),
    )
    if result.status == "blocked":
        pytest.skip("Pinned Aloy app dependencies are not installed")
    assert result.bundle is not None

    context = {
        "protocol_version": "1",
        "sdk_version": "1",
        "event_id": "event-smoke",
        "project_id": "project-smoke",
        "build_id": "build-smoke",
        "code_revision_id": "revision-smoke",
        "data_revision": 0,
        "capabilities": ["data:career"],
        "widgets": [],
        "data": {"interactions": [], "surface": {"career": []}},
    }
    assert (
        inspect_surface_runtime(
            build_surface_runtime_document(result.bundle),
            context,
            manifest=manifest,
        )
        == []
    )

    no_feedback_result = await LocalDevelopmentSurfaceBuildRunner().build(
        build_id="missing-command-feedback-local-toolchain",
        files={
            "/src/App.tsx": (
                'import React, { useState } from "react"; '
                'import { command } from "@aloy/surface"; '
                "export default function App(){const [company,setCompany]=useState('');"
                "return <form onSubmit={async event=>{event.preventDefault();await command("
                "'career.application_created',{applicationId:'smoke',company});}}>"
                "<label>Company<input aria-label='Company' value={company} "
                "onChange={event=>setCompany(event.target.value)}/></label>"
                "<button type='submit'>Save</button></form>}"
            )
        },
        manifest=manifest.model_dump(mode="json", by_alias=True),
    )
    assert no_feedback_result.bundle is not None
    no_feedback_diagnostics = inspect_surface_runtime(
        build_surface_runtime_document(no_feedback_result.bundle),
        context,
        manifest=manifest,
    )
    assert {item["code"] for item in no_feedback_diagnostics} == {
        "runtime_command_feedback_missing"
    }

    broken = SurfaceManifest.model_validate(
        manifest.model_dump(mode="json", by_alias=True)
    )
    broken_result = await LocalDevelopmentSurfaceBuildRunner().build(
        build_id="broken-interactive-local-toolchain",
        files={"/src/App.tsx": "export default () => <button>Save</button>"},
        manifest=broken.model_dump(mode="json", by_alias=True),
    )
    assert broken_result.bundle is not None
    diagnostics = inspect_surface_runtime(
        build_surface_runtime_document(broken_result.bundle),
        context,
        manifest=broken,
    )
    assert {item["code"] for item in diagnostics} == {"runtime_interaction_step_failed"}


def test_configured_surface_builder_requires_explicit_local_dev_mode(monkeypatch):
    from aloy_backend.config import settings

    monkeypatch.setattr(settings, "surface_build_backend", "local_dev")
    assert isinstance(
        configured_surface_build_runner(),
        LocalDevelopmentSurfaceBuildRunner,
    )


def _runtime_bundle(
    script: str = 'document.getElementById("root").textContent = "University";',
    style: str | None = "body { margin: 0; }",
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("surface.js", script)
        if style is not None:
            archive.writestr("surface.css", style)
    return output.getvalue()


async def _create_event(client, title: str = "University") -> dict:
    response = await client.post(
        "/v1/events",
        json={"title": title, "summary": "Persistent Event", "phase": "active"},
    )
    assert response.status_code == 201
    return response.json()


def _run_context(event_id: str, *, user_id: str = "test-user"):
    return authenticated_run_context(
        user_id=user_id,
        organization_id=f"user:{user_id}",
        run_id=f"run-build-{user_id}",
        session_id=f"session-build-{user_id}",
        event_id=event_id,
        workspace_id=event_id,
        agent_id="surface-builder",
    )


async def _author_revision(event_id: str, session_factory, *, content: str) -> str:
    handler = SurfaceAuthoringHandler(
        run_context=_run_context(event_id),
        session_factory=session_factory,
    )
    result = await handler.write(
        SurfaceWriteFilesParams(
            expected_revision=None,
            idempotency_key="author-source-0001",
            patches=[
                {
                    "path": "/workspace/src/App.tsx",
                    "content": content,
                }
            ],
        )
    )
    return result["draft"]["id"]


def test_surface_source_validation_is_deterministic_and_sdk_scoped():
    manifest = {"entrypoint": "/src/App.tsx", "sdk_version": "1"}
    valid = {
        "/src/App.tsx": (
            'import React from "react";\n'
            'import { intent } from "@aloy/surface";\n'
            "const parent = 'course';\n"
            "export default () => <main>{parent}</main>;\n"
        )
    }
    assert validate_surface_source(valid, manifest) == []

    invalid = {
        "/src/App.tsx": (
            'import Map from "unapproved-map-sdk";\n'
            "fetch('https://example.com');\n"
            "window.parent.postMessage('escape', '*');\n"
        ),
        "/src/styles.css": (
            '@import "https://example.com/theme.css";\n'
            "main { background: url(//example.com/image.png); }"
        ),
        "/src/node.ts": "const mode = process.env.NODE_ENV;",
    }
    diagnostics = validate_surface_source(invalid, manifest)
    assert {
        "undeclared_import",
        "direct_network",
        "host_escape",
        "css_import",
        "external_asset",
        "node_global",
    } <= {item["code"] for item in diagnostics}
    assert all(item["line"] is not None for item in diagnostics)
    assert diagnostics == validate_surface_source(invalid, manifest)


def test_interactive_surface_must_use_host_sdk_and_cannot_fake_bridge():
    manifest = {
        "entrypoint": "/src/App.tsx",
        "sdk_version": "1",
        "capabilities": ["event"],
    }
    diagnostics = validate_surface_source(
        {
            "/src/App.tsx": (
                "window.postMessage({ type: 'fake-ready' }, '*'); "
                "export default () => <main />"
            )
        },
        manifest,
    )
    assert {item["code"] for item in diagnostics} == {
        "direct_bridge",
        "missing_surface_sdk",
    }


def test_durable_surface_intents_require_checks_and_visible_sdk_failures():
    manifest = {
        "entrypoint": "/src/App.tsx",
        "sdk_version": "1",
        "capabilities": ["data:career"],
        "intents": {
            "career.application_created": {
                "class": "durable_selection",
                "schema": {"type": "object"},
                "write": {
                    "namespace": "career",
                    "key_field": "applicationId",
                },
            }
        },
    }
    diagnostics = validate_surface_source(
        {
            "/src/App.tsx": (
                'import { dispatch as hostDispatch } from "@aloy/surface"; '
                "export default () => <button onClick={() => "
                "hostDispatch('career.application_created', {})"
                ".catch(() => undefined)}>Save</button>"
            )
        },
        manifest,
    )
    assert {item["code"] for item in diagnostics} == {
        "missing_interaction_check",
        "swallowed_surface_failure",
    }


async def test_surface_build_retains_bundle_and_exposes_only_safe_metadata(
    client,
    db_session_maker,
    monkeypatch,
):
    event = await _create_event(client)
    revision_id = await _author_revision(
        event["id"],
        db_session_maker,
        content='import React from "react"; export default () => <main>Courses</main>',
    )
    bundle = _runtime_bundle(
        'document.getElementById("root").textContent = "</ScRiPt>University";',
        'body::after { content: "</StYlE>"; }',
    )
    runner = FakeBuildRunner(
        SurfaceBuildRunnerResult(
            status="succeeded",
            bundle=bundle,
            build_log="compiled cleanly",
            preview_artifacts=[
                {
                    "kind": "entry",
                    "name": "index.js",
                    "content_type": "text/javascript",
                    "sha256": "abc",
                    "size_bytes": "14",
                },
                {"kind": "bad-size", "size_bytes": "not-an-int"},
            ],
            resource_metrics={
                "duration_ms": 12,
                "runtime_inspection": "passed",
            },
        )
    )
    store = MemoryObjectStore()
    handler = SurfaceBuildHandler(
        run_context=_run_context(event["id"]),
        runner=runner,
        object_store=store,
        session_factory=db_session_maker,
    )

    built = await handler.build(
        SurfaceBuildParams(
            revision_id=revision_id,
            idempotency_key="build-university-0001",
        )
    )
    assert built["status"] == "succeeded", built
    assert built["bundle_available"] is True
    assert built["bundle_sha256"] == hashlib.sha256(bundle).hexdigest()
    assert built["bundle_size_bytes"] == len(bundle)
    assert built["preview_artifacts"][1]["size_bytes"] == 0
    assert "bundle_key" not in built

    replay = await handler.build(
        SurfaceBuildParams(
            revision_id=revision_id,
            idempotency_key="build-university-0001",
        )
    )
    assert replay["replayed"] is True
    assert len(runner.calls) == 1

    preview = await handler.preview(SurfacePreviewParams(build_id=built["id"]))
    assert preview["preview_ready"] is True
    assert preview["execution_available"] is True

    async with db_session_maker() as session:
        persisted = await session.get(SurfaceBuild, built["id"])
        assert persisted is not None
        assert persisted.bundle_key is not None
        with store.open(persisted.bundle_key) as retained:
            assert retained.read() == bundle
        trail = list(
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.event_id == event["id"],
                        EventTrailEntry.kind == "surface_build_finished",
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(trail) == 1
        assert trail[0].payload["status"] == "succeeded"

    listed = await client.get(f"/v1/events/{event['id']}/surface/builds")
    fetched = await client.get(f"/v1/events/{event['id']}/surface/builds/{built['id']}")
    denied = await client.get(
        f"/v1/events/{event['id']}/surface/builds/{built['id']}",
        headers={"X-Test-User": "other-user"},
    )
    assert listed.status_code == 200
    assert fetched.status_code == 200
    assert denied.status_code == 404
    for payload in [listed.json()[0], fetched.json()]:
        assert "bundle_key" not in payload
        assert "build_log" not in payload

    monkeypatch.setattr("aloy_backend.routes.surfaces.get_object_store", lambda: store)
    runtime = await client.get(
        f"/v1/events/{event['id']}/surface/builds/{built['id']}/runtime-document"
    )
    denied_runtime = await client.get(
        f"/v1/events/{event['id']}/surface/builds/{built['id']}/runtime-document",
        headers={"X-Test-User": "other-user"},
    )
    assert runtime.status_code == 200
    assert runtime.headers["cache-control"] == "private, no-store"
    assert runtime.headers["referrer-policy"] == "no-referrer"
    assert runtime.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'none'" in runtime.headers["content-security-policy"]
    assert "connect-src 'none'" in runtime.headers["content-security-policy"]
    assert "<\\/script>University" in runtime.text
    assert "<\\/style>" in runtime.text
    assert "bundle_key" not in runtime.text
    assert denied_runtime.status_code == 404

    async with db_session_maker() as session:
        persisted = await session.get(SurfaceBuild, built["id"])
        assert persisted is not None and persisted.bundle_key
        store.values[persisted.bundle_key] = b"not-a-zip"
    invalid_runtime = await client.get(
        f"/v1/events/{event['id']}/surface/builds/{built['id']}/runtime-document"
    )
    assert invalid_runtime.status_code == 409
    assert "valid ZIP" in invalid_runtime.json()["detail"]


def test_surface_runtime_rejects_non_contract_entries_and_missing_script():
    missing = io.BytesIO()
    with zipfile.ZipFile(missing, "w") as archive:
        archive.writestr("surface.css", "body {}")
    with pytest.raises(InvalidSurfaceBundle, match="missing surface.js"):
        build_surface_runtime_document(missing.getvalue())

    unknown = io.BytesIO()
    with zipfile.ZipFile(unknown, "w") as archive:
        archive.writestr("surface.js", "void 0")
        archive.writestr("index.html", "<h1>untrusted shell</h1>")
    with pytest.raises(InvalidSurfaceBundle, match="unsupported entries"):
        build_surface_runtime_document(unknown.getvalue())


async def test_surface_build_rejects_source_before_runner_execution(
    client,
    db_session_maker,
):
    event = await _create_event(client, "Madrid")
    revision_id = await _author_revision(
        event["id"],
        db_session_maker,
        content="export default () => { fetch('/secret'); return <main>Madrid</main> }",
    )
    runner = FakeBuildRunner(
        SurfaceBuildRunnerResult(status="succeeded", bundle=b"must-not-run")
    )
    handler = SurfaceBuildHandler(
        run_context=_run_context(event["id"]),
        runner=runner,
        object_store=MemoryObjectStore(),
        session_factory=db_session_maker,
    )

    result = await handler.build(
        SurfaceBuildParams(
            revision_id=revision_id,
            idempotency_key="build-madrid-0001",
        )
    )
    assert result["status"] == "failed"
    assert result["bundle_available"] is False
    assert {item["code"] for item in result["diagnostics"]} == {"direct_network"}
    assert runner.calls == []


async def test_preview_fails_closed_without_browser_inspection_receipt(
    client,
    db_session_maker,
):
    event = await _create_event(client, "Uninspected")
    revision_id = await _author_revision(
        event["id"],
        db_session_maker,
        content="export default () => <main>Uninspected</main>",
    )
    handler = SurfaceBuildHandler(
        run_context=_run_context(event["id"]),
        runner=FakeBuildRunner(
            SurfaceBuildRunnerResult(
                status="succeeded",
                bundle=_runtime_bundle(),
            )
        ),
        object_store=MemoryObjectStore(),
        session_factory=db_session_maker,
    )
    built = await handler.build(
        SurfaceBuildParams(
            revision_id=revision_id,
            idempotency_key="build-uninspected-0001",
        )
    )
    preview = await handler.preview(SurfacePreviewParams(build_id=built["id"]))

    assert preview["preview_ready"] is False
    assert preview["execution_available"] is False
    assert preview["runtime_diagnostics"][0]["code"] == (
        "runtime_inspector_unavailable"
    )


async def test_surface_build_is_blocked_without_isolation_and_rejects_large_bundle(
    client,
    db_session_maker,
):
    event = await _create_event(client, "Career OS")
    revision_id = await _author_revision(
        event["id"],
        db_session_maker,
        content="export default () => <main>Jobs</main>",
    )
    unavailable = SurfaceBuildHandler(
        run_context=_run_context(event["id"]),
        runner=UnavailableSurfaceBuildRunner(),
        session_factory=db_session_maker,
    )
    blocked = await unavailable.build(
        SurfaceBuildParams(
            revision_id=revision_id,
            idempotency_key="build-career-blocked-0001",
        )
    )
    assert blocked["status"] == "blocked"
    assert blocked["diagnostics"][0]["code"] == "isolated_builder_unavailable"

    oversized = SurfaceBuildHandler(
        run_context=_run_context(event["id"]),
        runner=FakeBuildRunner(
            SurfaceBuildRunnerResult(
                status="succeeded",
                bundle=b"x" * (MAX_SURFACE_BUNDLE_BYTES + 1),
            )
        ),
        object_store=MemoryObjectStore(),
        session_factory=db_session_maker,
    )
    failed = await oversized.build(
        SurfaceBuildParams(
            revision_id=revision_id,
            idempotency_key="build-career-large-0001",
        )
    )
    assert failed["status"] == "failed"
    assert failed["diagnostics"][0]["code"] == "bundle_too_large"
    assert failed["bundle_available"] is False

    with pytest.raises(ValueError, match="cannot build generated Surfaces"):
        SandboxSurfaceBuildRunner(LocalSandboxProvider())


async def test_surface_build_idempotency_key_cannot_cross_revisions(
    client,
    db_session_maker,
):
    event = await _create_event(client, "Timetable")
    author = SurfaceAuthoringHandler(
        run_context=_run_context(event["id"]),
        session_factory=db_session_maker,
    )
    first = await author.write(
        SurfaceWriteFilesParams(
            expected_revision=None,
            idempotency_key="author-timetable-0001",
            patches=[
                {
                    "path": "/workspace/src/App.tsx",
                    "content": "export default () => <main>One</main>",
                }
            ],
        )
    )
    second = await author.write(
        SurfaceWriteFilesParams(
            expected_revision=first["draft"]["id"],
            idempotency_key="author-timetable-0002",
            patches=[
                {
                    "path": "/workspace/src/App.tsx",
                    "content": "export default () => <main>Two</main>",
                }
            ],
        )
    )
    handler = SurfaceBuildHandler(
        run_context=_run_context(event["id"]),
        runner=FakeBuildRunner(
            SurfaceBuildRunnerResult(status="succeeded", bundle=b"bundle")
        ),
        object_store=MemoryObjectStore(),
        session_factory=db_session_maker,
    )
    await handler.build(
        SurfaceBuildParams(
            revision_id=first["draft"]["id"],
            idempotency_key="same-build-key-0001",
        )
    )
    with pytest.raises(SurfaceConflictError, match="different build"):
        await handler.build(
            SurfaceBuildParams(
                revision_id=second["draft"]["id"],
                idempotency_key="same-build-key-0001",
            )
        )


def test_surface_build_migration_creates_and_removes_build_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'surface-build-migration.db'}")
    metadata = sa.MetaData()
    sa.Table("events", metadata, sa.Column("id", sa.String(), primary_key=True))
    sa.Table(
        "surface_projects",
        metadata,
        sa.Column("id", sa.String(), primary_key=True),
    )
    sa.Table(
        "surface_revisions",
        metadata,
        sa.Column("id", sa.String(), primary_key=True),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        migration = importlib.import_module(
            "aloy_backend.alembic.versions.y1b2c3d4e5f6_surface_builds"
        )
        original_op = migration.op
        migration.op = Operations(MigrationContext.configure(connection))
        try:
            migration.upgrade()
            assert "surface_builds" in set(inspect(connection).get_table_names())
            columns = {
                column["name"]
                for column in inspect(connection).get_columns("surface_builds")
            }
            assert {
                "idempotency_key",
                "toolchain_version",
                "validation_result",
                "diagnostics",
                "bundle_key",
                "preview_artifacts",
            } <= columns
            migration.downgrade()
            assert "surface_builds" not in set(inspect(connection).get_table_names())
        finally:
            migration.op = original_op
    engine.dispose()
