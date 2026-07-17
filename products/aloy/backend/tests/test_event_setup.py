from __future__ import annotations

import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect
from sqlmodel import select

from aloy_backend.models import (
    Event,
    EventConnectionGrant,
    EventSetupContextItem,
    EventSetupDraft,
    EventTrailEntry,
    KnowledgeEntry,
    OAuthConnection,
    StoredFile,
)


async def test_event_setup_is_resumable_tenant_scoped_and_promotes_once(
    client, db_session_maker, monkeypatch, tmp_path
):
    from aloy_backend import config as config_mod
    from aloy_backend import storage as storage_mod

    monkeypatch.setattr(config_mod.settings, "storage_dir", str(tmp_path / "storage"))
    monkeypatch.setattr(storage_mod, "_STORE", None)

    assert (await client.get("/v1/event-drafts/current")).status_code == 204
    created = await client.post("/v1/event-drafts", json={})
    assert created.status_code == 201
    draft_id = created.json()["id"]
    assert created.json()["status"] == "open"

    updated = await client.patch(
        f"/v1/event-drafts/{draft_id}",
        json={
            "title": "University 2026",
            "description": "Computer science timetable, tests, and assignments.",
            "mode": "assisted",
        },
    )
    assert updated.status_code == 200
    resumed = await client.get("/v1/event-drafts/current")
    assert resumed.json()["id"] == draft_id
    assert resumed.json()["title"] == "University 2026"
    assert (
        await client.get(
            f"/v1/event-drafts/{draft_id}",
            headers={"X-Test-User": "someone-else"},
        )
    ).status_code == 404

    link = await client.post(
        f"/v1/event-drafts/{draft_id}/context",
        json={"kind": "link", "url": "https://university.example/timetable"},
    )
    assert link.status_code == 201
    assert link.json()["status"] == "pending"

    uploaded = await client.post(
        f"/v1/event-drafts/{draft_id}/files",
        files={"file": ("course-guide.pdf", b"course guide", "application/pdf")},
    )
    assert uploaded.status_code == 201
    assert uploaded.json()["status"] == "ready"

    async with db_session_maker() as session:
        session.add(
            OAuthConnection(
                organization_id="user:test-user",
                user_id="test-user",
                provider="google",
                scope="user",
                access_token_enc="encrypted-test-token",
                account_email="student@example.com",
            )
        )
        await session.commit()

    connected = await client.post(
        f"/v1/event-drafts/{draft_id}/context",
        json={
            "kind": "connection",
            "provider": "google",
            "connection_scope": "user",
        },
    )
    assert connected.status_code == 201
    assert connected.json()["metadata"]["provider"] == "google"
    assert "connection_id" not in connected.json()

    promoted = await client.post(f"/v1/event-drafts/{draft_id}/promote")
    assert promoted.status_code == 201
    event_id = promoted.json()["id"]
    assert promoted.json()["title"] == "University 2026"
    assert promoted.json()["conversation_id"]
    assert promoted.json()["cover"]["status"] == "queued"

    repeated = await client.post(f"/v1/event-drafts/{draft_id}/promote")
    assert repeated.status_code == 201
    assert repeated.json()["id"] == event_id
    assert (await client.get("/v1/event-drafts/current")).status_code == 204

    async with db_session_maker() as session:
        projects = list(
            (await session.execute(select(Event).where(Event.type == "project")))
            .scalars()
            .all()
        )
        assert [event.id for event in projects] == [event_id]

        draft = await session.get(EventSetupDraft, draft_id)
        assert draft is not None
        assert draft.status == "promoted"
        assert draft.created_event_id == event_id

        files = list(
            (
                await session.execute(
                    select(StoredFile).where(StoredFile.event_id == event_id)
                )
            )
            .scalars()
            .all()
        )
        assert [(item.name, item.kind, item.in_library) for item in files] == [
            ("course-guide.pdf", "upload", True)
        ]

        memories = list(
            (
                await session.execute(
                    select(KnowledgeEntry).where(KnowledgeEntry.event_id == event_id)
                )
            )
            .scalars()
            .all()
        )
        assert {tuple(item.tags or []) for item in memories} == {
            ("event-setup", "description"),
            ("event-setup", "link"),
        }
        assert all(item.metadata_["event_scoped"] for item in memories)

        grants = list(
            (
                await session.execute(
                    select(EventConnectionGrant).where(
                        EventConnectionGrant.event_id == event_id
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(grants) == 1
        assert grants[0].provider == "google"
        assert grants[0].access_scope == {"mode": "event", "resources": []}

        trail = list(
            (
                await session.execute(
                    select(EventTrailEntry).where(EventTrailEntry.event_id == event_id)
                )
            )
            .scalars()
            .all()
        )
        assert len(trail) == 1
        assert trail[0].kind == "event_created"
        assert trail[0].payload["context_count"] == 3

    monkeypatch.setattr(storage_mod, "_STORE", None)


async def test_pending_or_failed_context_never_blocks_event_creation(
    client, db_session_maker
):
    draft = await client.post(
        "/v1/event-drafts",
        json={"title": "Madrid trip", "description": "See El Clásico."},
    )
    draft_id = draft.json()["id"]
    link = await client.post(
        f"/v1/event-drafts/{draft_id}/context",
        json={"kind": "link", "url": "https://example.com/match"},
    )
    async with db_session_maker() as session:
        item = await session.get(EventSetupContextItem, link.json()["id"])
        assert item is not None
        item.status = "failed"
        item.error = "Source unavailable"
        session.add(item)
        await session.commit()

    promoted = await client.post(f"/v1/event-drafts/{draft_id}/promote")
    assert promoted.status_code == 201
    assert promoted.json()["title"] == "Madrid trip"


async def test_event_setup_validates_required_context_and_can_remove_it(client):
    draft = await client.post("/v1/event-drafts", json={"title": "Career OS"})
    draft_id = draft.json()["id"]
    assert (
        await client.post(f"/v1/event-drafts/{draft_id}/context", json={"kind": "link"})
    ).status_code == 422
    assert (
        await client.post(
            f"/v1/event-drafts/{draft_id}/context",
            json={"kind": "connection", "provider": "google"},
        )
    ).status_code == 404
    link = await client.post(
        f"/v1/event-drafts/{draft_id}/context",
        json={"kind": "link", "url": "https://example.com"},
    )
    removed = await client.delete(
        f"/v1/event-drafts/{draft_id}/context/{link.json()['id']}"
    )
    assert removed.status_code == 204
    resumed = await client.get(f"/v1/event-drafts/{draft_id}")
    assert resumed.json()["context_items"] == []


def test_event_setup_migration_round_trip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'event-setup.db'}")
    metadata = sa.MetaData()
    for table in ("events", "oauth_connections"):
        sa.Table(table, metadata, sa.Column("id", sa.String(), primary_key=True))
    metadata.create_all(engine)

    with engine.begin() as connection:
        migration = importlib.import_module(
            "aloy_backend.alembic.versions.b4e5f6a7c8d9_event_setup_context"
        )
        original_op = migration.op
        migration.op = Operations(MigrationContext.configure(connection))
        try:
            migration.upgrade()
            tables = set(inspect(connection).get_table_names())
            assert {
                "event_setup_drafts",
                "event_setup_context_items",
                "event_connection_grants",
            } <= tables
            draft_columns = {
                column["name"]
                for column in inspect(connection).get_columns("event_setup_drafts")
            }
            assert {
                "mode",
                "status",
                "description",
                "created_event_id",
            } <= draft_columns
            context_columns = {
                column["name"]
                for column in inspect(connection).get_columns(
                    "event_setup_context_items"
                )
            }
            assert {
                "kind",
                "status",
                "source_url",
                "connection_id",
                "storage_key",
                "metadata",
            } <= context_columns
            migration.downgrade()
            tables = set(inspect(connection).get_table_names())
            assert "event_setup_drafts" not in tables
            assert "event_setup_context_items" not in tables
            assert "event_connection_grants" not in tables
        finally:
            migration.op = original_op
    engine.dispose()
