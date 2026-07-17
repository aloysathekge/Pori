from __future__ import annotations

import importlib

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect
from sqlmodel import select

from aloy_backend.conversation_runtime import load_event_memory
from aloy_backend.event_context import (
    EventBriefPayload,
    EventEvidenceRef,
    GroundedText,
    publish_event_brief,
    refresh_event_context_snapshot,
)
from aloy_backend.models import (
    Conversation,
    Event,
    EventBrief,
    EventTrailEntry,
    KnowledgeEntry,
    Task,
)


async def test_context_snapshot_is_tenant_scoped_content_addressed_and_versioned(
    db_session_maker,
):
    async with db_session_maker() as session:
        event = Event(
            id="evt-context",
            organization_id="org-1",
            user_id="alice",
            title="University",
            summary="Manage the semester",
        )
        task = Task(
            id="task-context",
            organization_id="org-1",
            user_id="alice",
            event_id=event.id,
            title="Prepare for algorithms exam",
            created_by="alice",
        )
        session.add_all([event, task])
        await session.flush()

        first, first_pack, first_created = await refresh_event_context_snapshot(
            session,
            organization_id="org-1",
            user_id="alice",
            event_id=event.id,
        )
        same, same_pack, same_created = await refresh_event_context_snapshot(
            session,
            organization_id="org-1",
            user_id="alice",
            event_id=event.id,
        )

        assert first_created is True
        assert same_created is False
        assert same.id == first.id
        assert same_pack == first_pack
        assert first.version == 1
        assert first_pack.canonical_state["tasks"][0]["status"] == "open"

        task.status = "completed"
        session.add(task)
        await session.flush()
        changed, changed_pack, changed_created = await refresh_event_context_snapshot(
            session,
            organization_id="org-1",
            user_id="alice",
            event_id=event.id,
        )

        assert changed_created is True
        assert changed.version == 2
        assert changed.fingerprint != first.fingerprint
        assert changed_pack.canonical_state["tasks"][0]["status"] == "completed"

        with pytest.raises(ValueError, match="unavailable"):
            await refresh_event_context_snapshot(
                session,
                organization_id="other-org",
                user_id="alice",
                event_id=event.id,
            )


async def test_event_knowledge_overrides_conflicting_global_knowledge(
    db_session_maker,
):
    async with db_session_maker() as session:
        event = Event(
            id="evt-precedence",
            organization_id="org-1",
            user_id="alice",
            title="Trip",
        )
        conversation = Conversation(
            id="conv-precedence",
            organization_id="org-1",
            user_id="alice",
            event_id=event.id,
        )
        session.add_all(
            [
                event,
                conversation,
                KnowledgeEntry(
                    id="global-currency",
                    organization_id="org-1",
                    user_id="shared-org-row",
                    content="Use USD",
                    scope_level="org",
                    conflict_key="preferred-currency",
                ),
                KnowledgeEntry(
                    id="event-currency",
                    organization_id="org-1",
                    user_id="alice",
                    event_id=event.id,
                    content="Use EUR for this trip",
                    scope_level="personal",
                    conflict_key="preferred-currency",
                ),
            ]
        )
        await session.commit()

        memory = await load_event_memory(
            session,
            organization_id="org-1",
            user_id="alice",
            conversation=conversation,
        )

        assert {item.content for item in memory.memory_records} == {
            "Use EUR for this trip"
        }
        assert memory.trusted_context is not None
        assert memory.trusted_context_fingerprint


async def test_event_brief_requires_readiness_and_known_evidence_and_is_idempotent(
    db_session_maker,
):
    async with db_session_maker() as session:
        event = Event(
            id="evt-brief",
            organization_id="org-1",
            user_id="alice",
            title="University",
        )
        session.add(event)
        await session.flush()
        weak_snapshot, _, _ = await refresh_event_context_snapshot(
            session,
            organization_id="org-1",
            user_id="alice",
            event_id=event.id,
        )
        weak_payload = EventBriefPayload(
            purpose=GroundedText(
                text="Manage university",
                evidence_refs=[EventEvidenceRef(kind="knowledge_entry", id="missing")],
            )
        )
        with pytest.raises(ValueError, match="not ready"):
            await publish_event_brief(
                session,
                organization_id="org-1",
                user_id="alice",
                event_id=event.id,
                snapshot_id=weak_snapshot.id,
                payload=weak_payload,
            )

        evidence = KnowledgeEntry(
            id="evidence-university",
            organization_id="org-1",
            user_id="alice",
            event_id=event.id,
            content=(
                "Course calendar, exam dates, study goals, and constraints. " * 15
            ),
            sensitivity="confidential",
        )
        session.add(evidence)
        await session.flush()
        ready_snapshot, ready_pack, _ = await refresh_event_context_snapshot(
            session,
            organization_id="org-1",
            user_id="alice",
            event_id=event.id,
        )
        assert ready_pack.readiness.level == "sufficient"
        assert ready_snapshot.provider_cache_allowed is False

        # A same-length evidence correction must still invalidate the snapshot;
        # timestamps and character counts alone are not content identity.
        previous_content = evidence.content
        evidence.content = "X" + previous_content[1:]
        session.add(evidence)
        await session.flush()
        corrected_snapshot, _, corrected = await refresh_event_context_snapshot(
            session,
            organization_id="org-1",
            user_id="alice",
            event_id=event.id,
        )
        assert corrected is True
        assert corrected_snapshot.fingerprint != ready_snapshot.fingerprint
        ready_snapshot = corrected_snapshot

        payload = EventBriefPayload(
            purpose=GroundedText(
                text="Keep the semester on track",
                evidence_refs=[
                    EventEvidenceRef(kind="knowledge_entry", id="evidence-university")
                ],
            ),
            unknowns=["Exact office hours are unknown."],
        )
        brief, created = await publish_event_brief(
            session,
            organization_id="org-1",
            user_id="alice",
            event_id=event.id,
            snapshot_id=ready_snapshot.id,
            payload=payload,
            creator_run_id="run-bootstrap",
        )
        same, created_again = await publish_event_brief(
            session,
            organization_id="org-1",
            user_id="alice",
            event_id=event.id,
            snapshot_id=ready_snapshot.id,
            payload=payload,
            creator_run_id="run-bootstrap",
        )

        assert created is True
        assert created_again is False
        assert same.id == brief.id
        assert brief.version == 1
        assert brief.evidence_refs == [
            {"kind": "knowledge_entry", "id": "evidence-university"}
        ]
        trail = (
            await session.execute(
                select(EventTrailEntry).where(
                    EventTrailEntry.event_id == event.id,
                    EventTrailEntry.kind == "event_brief_published",
                )
            )
        ).scalar_one()
        assert trail.run_id == "run-bootstrap"
        assert "context_snapshot_id" in trail.evidence_refs[1]
        assert (
            len(
                list(
                    (
                        await session.execute(
                            select(EventBrief).where(EventBrief.event_id == event.id)
                        )
                    )
                    .scalars()
                    .all()
                )
            )
            == 1
        )


async def test_event_surface_exposes_context_readiness_without_evidence_bodies(
    client,
):
    created = await client.post(
        "/v1/events",
        json={
            "title": "Career OS",
            "summary": "Research target companies, roles, constraints, and application strategy. "
            * 10,
        },
    )
    response = await client.get(f"/v1/events/{created.json()['id']}")

    assert response.status_code == 200
    status_section = next(
        section
        for section in response.json()["surface"]["sections"]
        if section["kind"] == "context_status"
    )
    status = status_section["status"]
    assert status["snapshot_version"] == 1
    assert status["readiness"]["level"] == "sufficient"
    assert status["readiness"]["should_bootstrap"] is True
    assert "evidence_catalog" not in status
    assert "pack" not in status

    second = await client.get(f"/v1/events/{created.json()['id']}")
    second_status = next(
        section["status"]
        for section in second.json()["surface"]["sections"]
        if section["kind"] == "context_status"
    )
    assert second_status["snapshot_id"] == status["snapshot_id"]
    assert second_status["fingerprint"] == status["fingerprint"]

    # The surface read created exactly one content-addressed snapshot.
    assert second_status["snapshot_version"] == 1


def test_event_context_migration_round_trip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'event-context.db'}")
    metadata = sa.MetaData()
    sa.Table("events", metadata, sa.Column("id", sa.String(), primary_key=True))
    metadata.create_all(engine)

    with engine.begin() as connection:
        migration = importlib.import_module(
            "aloy_backend.alembic.versions.d6a7b8c9e0f1_event_context_snapshots"
        )
        original_op = migration.op
        migration.op = Operations(MigrationContext.configure(connection))
        try:
            migration.upgrade()
            tables = set(inspect(connection).get_table_names())
            assert {"event_context_snapshots", "event_briefs"} <= tables
            brief_columns = {
                column["name"]
                for column in inspect(connection).get_columns("event_briefs")
            }
            assert {
                "source_context_snapshot_id",
                "fingerprint",
                "payload",
                "evidence_refs",
            } <= brief_columns

            migration.downgrade()
            tables = set(inspect(connection).get_table_names())
            assert "event_context_snapshots" not in tables
            assert "event_briefs" not in tables
        finally:
            migration.op = original_op
    engine.dispose()
