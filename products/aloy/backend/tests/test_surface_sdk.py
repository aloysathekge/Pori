from __future__ import annotations

import importlib
from types import SimpleNamespace

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect
from sqlmodel import col, func, select

import aloy_backend.proposal_executor as executor_module
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
from aloy_backend.proposal_executor import decide_proposal, execute_proposal
from aloy_backend.surface_lifecycle import (
    mark_surface_run_started,
    reconcile_surface_run,
)
from aloy_backend.surface_manifest import SurfaceManifest, validate_intent_payload
from aloy_backend.tools.gmail import GmailSendParams
from pori.tools.registry import ToolRegistry


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
            validation_result={"passed": True},
            diagnostics=[],
            bundle_key="test/surface.zip",
            bundle_sha256="abc123",
            bundle_size_bytes=100,
        )
        project.published_revision_id = revision.id
        project.published_build_id = build.id
        project.lifecycle = "published"
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


def _action_manifest() -> dict:
    return SurfaceManifest.model_validate(
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


def _action_request(build_id: str, revision_id: str, *, key: str) -> dict:
    return {
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
        "idempotency_key": key,
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
    assert set(context["data"]) == {"event", "tasks", "interactions", "surface"}
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
    assert refreshed.json()["data"]["interactions"][0]["status"] == "committed"

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

        run = runs[0]
        run.status = "running"
        session.add(run)
        assert await mark_surface_run_started(session, run=run) is True
        await session.commit()

    async with db_session_maker() as session:
        run = (await session.execute(select(Run))).scalars().one()
        run.status = "completed"
        run.success = True
        run.steps_taken = 3
        run.final_answer = "Hotel h2 is the stronger match-weekend option."
        outcome = Message(
            conversation_id=event["conversation_id"],
            role="assistant",
            content=run.final_answer,
            metadata_={"run_id": run.id},
        )
        session.add(run)
        session.add(outcome)
        interaction = await reconcile_surface_run(
            session,
            run=run,
            outcome_message=outcome,
        )
        assert interaction is not None
        await session.commit()

    async with db_session_maker() as session:
        run = (await session.execute(select(Run))).scalars().one()
        await reconcile_surface_run(session, run=run)
        await session.commit()
        interaction = (
            (await session.execute(select(SurfaceInteraction))).scalars().one()
        )
        messages = list((await session.execute(select(Message))).scalars().all())
        lifecycle = list(
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        col(EventTrailEntry.kind).in_(
                            [
                                "surface_reasoning_started",
                                "surface_reasoning_completed",
                            ]
                        )
                    )
                )
            )
            .scalars()
            .all()
        )
        assert interaction.status == "completed"
        assert interaction.outcome_message_id == outcome.id
        assert len(messages) == 2
        assert messages[-1].metadata_["kind"] == "surface_reasoning_result"
        assert len(lifecycle) == 2


async def test_surface_external_action_stages_proposal_without_execution(
    client,
    db_session_maker,
):
    event = await _create_event(client, "Career OS")
    build_id, revision_id = await _seed_runtime(
        db_session_maker, event["id"], _action_manifest()
    )
    response = await client.post(
        f"/v1/events/{event['id']}/surface/interactions",
        json=_action_request(build_id, revision_id, key="career-email-summary-0001"),
    )
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "waiting_approval"
    assert response.json()["proposal_id"]
    async with db_session_maker() as session:
        proposal = (await session.execute(select(ActionProposal))).scalars().one()
        assert proposal.tool == "gmail_send"
        assert proposal.status == "pending"
        assert proposal.receipt is None
        interaction = (
            (await session.execute(select(SurfaceInteraction))).scalars().one()
        )
        request_message = await session.get(Message, interaction.request_message_id)
        assert request_message.metadata_["kind"] == "surface_action_lifecycle"
        assert request_message.metadata_["phase"] == "request"

    rejected = await client.post(
        f"/v1/events/{event['id']}/proposals/{response.json()['proposal_id']}/decision",
        json={"decision": "reject"},
    )
    repeated = await client.post(
        f"/v1/events/{event['id']}/proposals/{response.json()['proposal_id']}/decision",
        json={"decision": "reject"},
    )
    assert rejected.status_code == 200
    assert repeated.status_code == 409
    async with db_session_maker() as session:
        interaction = (
            (await session.execute(select(SurfaceInteraction))).scalars().one()
        )
        cards = list(
            (
                await session.execute(
                    select(Message).where(
                        Message.conversation_id == event["conversation_id"],
                        Message.role == "assistant",
                    )
                )
            )
            .scalars()
            .all()
        )
        assert interaction.status == "rejected"
        assert interaction.outcome_message_id is not None
        assert [card.metadata_["phase"] for card in cards] == ["request", "outcome"]


async def test_surface_action_execution_reconciles_receipt_exactly_once(
    client,
    db_session_maker,
    monkeypatch,
):
    event = await _create_event(client, "Career OS")
    build_id, revision_id = await _seed_runtime(
        db_session_maker, event["id"], _action_manifest()
    )
    staged = await client.post(
        f"/v1/events/{event['id']}/surface/interactions",
        json=_action_request(
            build_id, revision_id, key="career-email-summary-commit-0001"
        ),
    )
    assert staged.status_code == 202

    calls: list[dict] = []

    def send(params: GmailSendParams, context: dict) -> dict:
        calls.append({"to": params.to, "attempt": context["execution_attempt_id"]})
        return {"sent": True, "id": "provider-message-surface", "to": params.to}

    registry = ToolRegistry()
    registry.register_tool("gmail_send", GmailSendParams, send, "send")

    async def available(*args, **kwargs):
        return SimpleNamespace(
            denied_tools=(),
            tool_context_extra={"connections": {"google": {"access_token": "test"}}},
        )

    monkeypatch.setattr(executor_module, "resolve_run_surface", available)
    async with db_session_maker() as session:
        proposal = await session.get(ActionProposal, staged.json()["proposal_id"])
        decided = await decide_proposal(
            session,
            event_id=event["id"],
            proposal_id=proposal.id,
            organization_id=proposal.organization_id,
            user_id=proposal.user_id,
            actor_id=proposal.user_id,
            decision="approve",
            registry=registry,
        )
        assert decided.status == "approved"

    first = await execute_proposal(
        staged.json()["proposal_id"],
        session_factory=db_session_maker,
        registry=registry,
    )
    second = await execute_proposal(
        staged.json()["proposal_id"],
        session_factory=db_session_maker,
        registry=registry,
    )
    assert first.status == "committed"
    assert second.claimed is False
    assert len(calls) == 1

    async with db_session_maker() as session:
        interaction = (
            (await session.execute(select(SurfaceInteraction))).scalars().one()
        )
        outcome_cards = list(
            (
                await session.execute(
                    select(Message).where(
                        Message.conversation_id == event["conversation_id"]
                    )
                )
            )
            .scalars()
            .all()
        )
        assert interaction.status == "committed"
        assert interaction.result["receipt"]["status"] == "succeeded"
        assert interaction.result["provider_operation_id"] == (
            "provider-message-surface"
        )
        assert len(outcome_cards) == 2
        assert outcome_cards[-1].metadata_["status"] == "committed"


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
            assert {"request_message_id", "outcome_message_id"} <= {
                column["name"]
                for column in inspect(connection).get_columns("surface_interactions")
            }
            migration.downgrade()
            assert "surface_interactions" not in set(
                inspect(connection).get_table_names()
            )
        finally:
            migration.op = original_op
    engine.dispose()
