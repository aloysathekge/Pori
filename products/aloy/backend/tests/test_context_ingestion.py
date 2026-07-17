from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect
from sqlmodel import select

from aloy_backend.context_ingestion import (
    ContextIngestionError,
    IngestedSource,
    claim_next_context_item,
    execute_claimed_context_item,
)
from aloy_backend.models import EventSetupContextItem, EventTrailEntry, KnowledgeEntry


async def _draft_with_context(client, *, kind: str, value: str) -> tuple[str, str]:
    draft = await client.post(
        "/v1/event-drafts",
        json={"title": "Research workspace", "description": "Trusted setup context."},
    )
    draft_id = draft.json()["id"]
    if kind == "link":
        added = await client.post(
            f"/v1/event-drafts/{draft_id}/context",
            json={"kind": "link", "url": value},
        )
    else:
        content_type = (
            "application/octet-stream" if value.endswith(".bin") else "text/plain"
        )
        added = await client.post(
            f"/v1/event-drafts/{draft_id}/files",
            files={
                "file": (
                    value,
                    b"Course: Algorithms\nExam: 24 October",
                    content_type,
                )
            },
        )
    promoted = await client.post(f"/v1/event-drafts/{draft_id}/promote")
    return promoted.json()["id"], added.json()["id"]


async def test_file_context_is_leased_ingested_and_event_scoped(
    client, db_session_maker, monkeypatch, tmp_path
):
    from aloy_backend import config as config_mod
    from aloy_backend import context_ingestion as ingestion_mod
    from aloy_backend import storage as storage_mod

    monkeypatch.setattr(config_mod.settings, "storage_dir", str(tmp_path / "storage"))
    monkeypatch.setattr(storage_mod, "_STORE", None)
    monkeypatch.setattr(ingestion_mod, "async_session", db_session_maker)

    event_id, item_id = await _draft_with_context(
        client, kind="file", value="course-notes.txt"
    )
    assert await claim_next_context_item("worker-a") == item_id
    assert await execute_claimed_context_item(item_id, "worker-a") is True
    assert await execute_claimed_context_item(item_id, "worker-a") is False

    async with db_session_maker() as session:
        item = await session.get(EventSetupContextItem, item_id)
        assert item is not None
        assert item.status == "ready"
        assert item.event_id == event_id
        assert item.attempt_count == 1
        assert item.lease_owner is None
        assert item.ingested_at is not None

        entry = await session.get(KnowledgeEntry, item.knowledge_entry_id)
        assert entry is not None
        assert entry.event_id == event_id
        assert "Course: Algorithms" in entry.content
        assert entry.tags == ["event-context", "file", "ingested"]
        assert entry.sensitivity == "internal"
        assert entry.retention == {"mode": "event_lifecycle"}
        assert entry.provenance["context_item_id"] == item_id

        completed = list(
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.event_id == event_id,
                        EventTrailEntry.kind == "context_ingestion_completed",
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(completed) == 1

    surface = await client.get(f"/v1/events/{event_id}")
    context = next(
        section
        for section in surface.json()["surface"]["sections"]
        if section["kind"] == "context"
    )
    assert context["items"][0]["status"] == "ready"
    assert "knowledge_entry_id" not in context["items"][0]
    monkeypatch.setattr(storage_mod, "_STORE", None)


async def test_link_context_reuses_placeholder_knowledge_and_records_freshness(
    client, db_session_maker, monkeypatch
):
    from aloy_backend import context_ingestion as ingestion_mod

    monkeypatch.setattr(ingestion_mod, "async_session", db_session_maker)
    event_id, item_id = await _draft_with_context(
        client, kind="link", value="https://example.com/timetable"
    )

    async def fetcher(url: str) -> IngestedSource:
        assert url == "https://example.com/timetable"
        return IngestedSource(
            text="Monday: Databases at 09:00",
            title="Semester timetable",
            content_type="text/html",
            retrieved_at=datetime(2026, 7, 17, 8, 0, tzinfo=timezone.utc),
            sha256="content-sha",
            metadata={"final_url": url, "etag": '"v1"'},
        )

    assert await claim_next_context_item("worker-link") == item_id
    assert (
        await execute_claimed_context_item(item_id, "worker-link", link_fetcher=fetcher)
        is True
    )

    async with db_session_maker() as session:
        memories = list(
            (
                await session.execute(
                    select(KnowledgeEntry).where(KnowledgeEntry.event_id == event_id)
                )
            )
            .scalars()
            .all()
        )
        links = [entry for entry in memories if "link" in (entry.tags or [])]
        assert len(links) == 1
        assert "Monday: Databases" in links[0].content
        assert links[0].provenance["freshness"]["etag"] == '"v1"'


async def test_expired_lease_is_recovered_by_another_worker(
    client, db_session_maker, monkeypatch
):
    from aloy_backend import context_ingestion as ingestion_mod

    monkeypatch.setattr(ingestion_mod, "async_session", db_session_maker)
    _, item_id = await _draft_with_context(
        client, kind="link", value="https://example.com/source"
    )
    async with db_session_maker() as session:
        item = await session.get(EventSetupContextItem, item_id)
        assert item is not None
        item.status = "ingesting"
        item.attempt_count = 1
        item.lease_owner = "dead-worker"
        item.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.add(item)
        await session.commit()

    assert await claim_next_context_item("worker-b") == item_id
    async with db_session_maker() as session:
        recovered = await session.get(EventSetupContextItem, item_id)
        assert recovered is not None
        assert recovered.status == "ingesting"
        assert recovered.attempt_count == 2
        assert recovered.lease_owner == "worker-b"


async def test_transient_failure_schedules_retry_and_manual_retry_is_tenant_scoped(
    client, db_session_maker, monkeypatch
):
    from aloy_backend import context_ingestion as ingestion_mod

    monkeypatch.setattr(ingestion_mod, "async_session", db_session_maker)
    event_id, item_id = await _draft_with_context(
        client, kind="link", value="https://example.com/unavailable"
    )

    async def unavailable(_: str) -> IngestedSource:
        raise ContextIngestionError("Temporary upstream failure")

    assert await claim_next_context_item("worker-a") == item_id
    assert (
        await execute_claimed_context_item(
            item_id, "worker-a", link_fetcher=unavailable
        )
        is True
    )
    async with db_session_maker() as session:
        failed = await session.get(EventSetupContextItem, item_id)
        assert failed is not None
        assert failed.status == "failed"
        assert failed.attempt_count == 1
        assert failed.next_attempt_at is not None
        assert failed.lease_owner is None

    assert await claim_next_context_item("too-early") is None
    denied = await client.post(
        f"/v1/events/{event_id}/context/{item_id}/retry",
        headers={"X-Test-User": "someone-else"},
    )
    assert denied.status_code == 404
    retried = await client.post(f"/v1/events/{event_id}/context/{item_id}/retry")
    assert retried.status_code == 202
    assert retried.json()["status"] == "pending"
    assert retried.json()["attempt_count"] == 0
    assert await claim_next_context_item("worker-b") == item_id


async def test_unsupported_file_fails_permanently(
    client, db_session_maker, monkeypatch, tmp_path
):
    from aloy_backend import config as config_mod
    from aloy_backend import context_ingestion as ingestion_mod
    from aloy_backend import storage as storage_mod

    monkeypatch.setattr(config_mod.settings, "storage_dir", str(tmp_path / "storage"))
    monkeypatch.setattr(storage_mod, "_STORE", None)
    monkeypatch.setattr(ingestion_mod, "async_session", db_session_maker)
    event_id, item_id = await _draft_with_context(
        client, kind="file", value="archive.bin"
    )
    assert await claim_next_context_item("worker-a") == item_id
    assert await execute_claimed_context_item(item_id, "worker-a") is True

    async with db_session_maker() as session:
        item = await session.get(EventSetupContextItem, item_id)
        assert item is not None
        assert item.status == "failed"
        assert item.attempt_count == item.max_attempts
        assert item.next_attempt_at is None
        trail = list(
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.event_id == event_id,
                        EventTrailEntry.kind == "context_ingestion_failed",
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(trail) == 1
        assert trail[0].payload["retryable"] is False

    monkeypatch.setattr(storage_mod, "_STORE", None)


def test_context_ingestion_migration_round_trip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'context-ingestion.db'}")
    metadata = sa.MetaData()
    sa.Table("events", metadata, sa.Column("id", sa.String(), primary_key=True))
    sa.Table(
        "event_setup_context_items",
        metadata,
        sa.Column("id", sa.String(), primary_key=True),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        migration = importlib.import_module(
            "aloy_backend.alembic.versions.c5f6a7b8d9e0_event_context_ingestion"
        )
        original_op = migration.op
        migration.op = Operations(MigrationContext.configure(connection))
        try:
            migration.upgrade()
            columns = {
                column["name"]
                for column in inspect(connection).get_columns(
                    "event_setup_context_items"
                )
            }
            assert {
                "event_id",
                "sensitivity",
                "attempt_count",
                "lease_owner",
                "lease_expires_at",
                "next_attempt_at",
                "knowledge_entry_id",
            } <= columns
            indexes = {
                index["name"]
                for index in inspect(connection).get_indexes(
                    "event_setup_context_items"
                )
            }
            assert "ix_event_setup_context_items_event_id" in indexes
            migration.downgrade()
            assert {
                column["name"]
                for column in inspect(connection).get_columns(
                    "event_setup_context_items"
                )
            } == {"id"}
        finally:
            migration.op = original_op
    engine.dispose()
