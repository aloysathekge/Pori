from __future__ import annotations

import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect
from sqlmodel import func, select

from aloy_backend.models import (
    ActionProposal,
    Event,
    EventTrailEntry,
    Message,
    Run,
    SurfaceBuild,
    SurfaceDataRecord,
    SurfaceInteraction,
    SurfaceProject,
    SurfaceRevision,
)
from aloy_backend.surface_manifest import SurfaceManifest, validate_intent_payload


async def _create_event(client, title: str = "University") -> dict:
    response = await client.post(
        "/v1/events",
        json={"title": title, "summary": "Persistent Event", "phase": "active"},
    )
    assert response.status_code == 201
    return response.json()


async def _seed_runtime(
    db_session_maker, event_id: str, manifest: dict
) -> tuple[str, str]:
    async with db_session_maker() as session:
        event = await session.get(Event, event_id)
        assert event is not None
        project = SurfaceProject(
            organization_id=event.organization_id,
            user_id=event.user_id,
            event_id=event.id,
        )
        session.add(project)
        await session.flush()
        revision = SurfaceRevision(
            organization_id=event.organization_id,
            user_id=event.user_id,
            event_id=event.id,
            project_id=project.id,
            revision_number=1,
            idempotency_key="sdk-author-0001",
            request_fingerprint="author-fingerprint",
            manifest=manifest,
            files={"/src/App.tsx": "export default () => null"},
            checksum="source-checksum",
            file_count=1,
            total_bytes=27,
        )
        session.add(revision)
        await session.flush()
        project.draft_revision_id = revision.id
        build = SurfaceBuild(
            organization_id=event.organization_id,
            user_id=event.user_id,
            event_id=event.id,
            project_id=project.id,
            revision_id=revision.id,
            idempotency_key="sdk-build-0001",
            request_fingerprint="build-fingerprint",
            status="succeeded",
            source_checksum=revision.checksum,
            toolchain_version="aloy-surface-toolchain@1",
            validation_result={"valid": True},
            diagnostics=[],
            bundle_key="test/surface.zip",
            bundle_sha256="abc123",
            bundle_size_bytes=100,
        )
        session.add(project)
        session.add(build)
        await session.commit()
        return build.id, revision.id


def _selection_manifest() -> dict:
    return {
        "format": "aloy-react-surface",
        "entrypoint": "/src/App.tsx",
        "sdk_version": "1",
        "capabilities": ["event", "tasks", "data:academic"],
        "intents": {
            "academic.course_selected": {
                "class": "durable_selection",
                "schema": {
                    "type": "object",
                    "properties": {"courseId": {"type": "string", "maxLength": 40}},
                    "required": ["courseId"],
                    "additionalProperties": False,
                },
                "write": {
                    "namespace": "academic",
                    "key_field": "courseId",
                    "posture": "user_reported",
                },
            }
        },
        "widgets": [],
    }


async def test_surface_context_and_durable_dispatch_are_capability_scoped_and_exactly_once(
    client,
    db_session_maker,
):
    event = await _create_event(client)
    task = await client.post(
        f"/v1/events/{event['id']}/tasks", json={"title": "Study MAT204"}
    )
    assert task.status_code == 201
    build_id, revision_id = await _seed_runtime(
        db_session_maker, event["id"], _selection_manifest()
    )

    initial = await client.get(
        f"/v1/events/{event['id']}/surface/context", params={"build_id": build_id}
    )
    denied = await client.get(
        f"/v1/events/{event['id']}/surface/context",
        params={"build_id": build_id},
        headers={"X-Test-User": "other-user"},
    )
    assert initial.status_code == 200
    context = initial.json()
    assert context["code_revision_id"] == revision_id
    assert context["data_revision"] == 0
    assert context["data"]["event"]["title"] == "University"
    assert context["data"]["tasks"][0]["title"] == "Study MAT204"
    assert set(context["data"]) == {"event", "tasks", "surface"}
    assert denied.status_code == 404

    body = {
        "build_id": build_id,
        "code_revision_id": revision_id,
        "data_revision": 0,
        "method": "dispatch",
        "name": "academic.course_selected",
        "component_id": "course-card-mat204",
        "payload": {"courseId": "MAT204"},
        "idempotency_key": "course-select-mat204-0001",
    }
    created = await client.post(
        f"/v1/events/{event['id']}/surface/interactions", json=body
    )
    replay = await client.post(
        f"/v1/events/{event['id']}/surface/interactions", json=body
    )
    conflicting = await client.post(
        f"/v1/events/{event['id']}/surface/interactions",
        json={**body, "payload": {"courseId": "PHY201"}},
    )
    stale = await client.post(
        f"/v1/events/{event['id']}/surface/interactions",
        json={**body, "idempotency_key": "course-select-stale-0002"},
    )
    assert created.status_code == 202, created.text
    assert created.json()["status"] == "committed"
    assert created.json()["data_revision"] == 1
    assert replay.status_code == 202
    assert replay.json()["replayed"] is True
    assert conflicting.status_code == 409
    assert stale.status_code == 409

    refreshed = await client.get(
        f"/v1/events/{event['id']}/surface/context", params={"build_id": build_id}
    )
    record = refreshed.json()["data"]["surface"]["academic"][0]
    assert refreshed.json()["data_revision"] == 1
    assert record["key"] == "MAT204"
    assert record["posture"] == "user_reported"

    async with db_session_maker() as session:
        assert (
            await session.execute(select(func.count()).select_from(SurfaceInteraction))
        ).scalar_one() == 1
        assert (
            await session.execute(select(func.count()).select_from(SurfaceDataRecord))
        ).scalar_one() == 1
        trail = list(
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.kind == "surface_interaction_committed"
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(trail) == 1


async def test_surface_ask_aloy_queues_one_canonical_conversation_run(
    client,
    db_session_maker,
):
    event = await _create_event(client, "Madrid")
    manifest = SurfaceManifest(capabilities=["ask_aloy"]).model_dump(
        mode="json", by_alias=True
    )
    build_id, revision_id = await _seed_runtime(db_session_maker, event["id"], manifest)
    body = {
        "build_id": build_id,
        "code_revision_id": revision_id,
        "data_revision": 0,
        "method": "ask_aloy",
        "name": "aloy.ask",
        "component_id": "hotel-comparison",
        "payload": {"hotelIds": ["h1", "h2"]},
        "message": "Compare these two hotels for the match weekend.",
        "idempotency_key": "madrid-compare-hotels-0001",
    }
    created = await client.post(
        f"/v1/events/{event['id']}/surface/interactions", json=body
    )
    replay = await client.post(
        f"/v1/events/{event['id']}/surface/interactions", json=body
    )
    assert created.status_code == 202, created.text
    assert created.json()["status"] == "queued"
    assert created.json()["handling_run_id"]
    assert replay.json()["replayed"] is True

    async with db_session_maker() as session:
        runs = list((await session.execute(select(Run))).scalars().all())
        messages = list(
            (await session.execute(select(Message).where(Message.role == "user")))
            .scalars()
            .all()
        )
        assert len(runs) == 1
        assert runs[0].conversation_id == event["conversation_id"]
        assert len(messages) == 1
        assert messages[0].metadata_["kind"] == "surface_interaction"


async def test_surface_external_action_stages_proposal_without_execution(
    client,
    db_session_maker,
):
    event = await _create_event(client, "Career OS")
    manifest = SurfaceManifest.model_validate(
        {
            "capabilities": ["proposals"],
            "intents": {
                "career.email_summary": {
                    "class": "external_action",
                    "tool": "gmail_send",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["to", "subject", "body"],
                        "additionalProperties": False,
                    },
                }
            },
        }
    ).model_dump(mode="json", by_alias=True)
    build_id, revision_id = await _seed_runtime(db_session_maker, event["id"], manifest)
    response = await client.post(
        f"/v1/events/{event['id']}/surface/interactions",
        json={
            "build_id": build_id,
            "code_revision_id": revision_id,
            "data_revision": 0,
            "method": "request_action",
            "name": "career.email_summary",
            "component_id": "send-summary",
            "payload": {
                "to": "founder@example.com",
                "subject": "Startup roles",
                "body": "Here are the shortlisted companies.",
            },
            "reason": "Send the approved research summary.",
            "idempotency_key": "career-email-summary-0001",
        },
    )
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "waiting_approval"
    assert response.json()["proposal_id"]
    async with db_session_maker() as session:
        proposal = (await session.execute(select(ActionProposal))).scalars().one()
        assert proposal.tool == "gmail_send"
        assert proposal.status == "pending"
        assert proposal.receipt is None


def test_surface_manifest_and_payload_validation_fail_closed():
    schema = {
        "type": "object",
        "properties": {"courseId": {"type": "string"}},
        "required": ["courseId"],
        "additionalProperties": False,
    }
    validate_intent_payload(schema, {"courseId": "MAT204"})
    for bad in ({}, {"courseId": 204}, {"courseId": "MAT204", "admin": True}):
        try:
            validate_intent_payload(schema, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Payload should have failed: {bad}")


def test_surface_sdk_migration_creates_and_removes_tables(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'surface-sdk-migration.db'}")
    metadata = sa.MetaData()
    for table in ("events", "surface_projects", "surface_revisions", "surface_builds"):
        columns = [sa.Column("id", sa.String(), primary_key=True)]
        if table == "surface_projects":
            columns.append(sa.Column("updated_at", sa.DateTime(timezone=True)))
        sa.Table(table, metadata, *columns)
    metadata.create_all(engine)
    with engine.begin() as connection:
        migration = importlib.import_module(
            "aloy_backend.alembic.versions.z2c3d4e5f6a7_surface_sdk_data_interactions"
        )
        original_op = migration.op
        migration.op = Operations(MigrationContext.configure(connection))
        try:
            migration.upgrade()
            tables = set(inspect(connection).get_table_names())
            assert {"surface_data_records", "surface_interactions"} <= tables
            assert "data_revision" in {
                column["name"]
                for column in inspect(connection).get_columns("surface_projects")
            }
            migration.downgrade()
            assert "surface_interactions" not in set(
                inspect(connection).get_table_names()
            )
        finally:
            migration.op = original_op
    engine.dispose()
