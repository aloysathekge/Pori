from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect
from sqlmodel import select

from aloy_backend import background as background_module
from aloy_backend import event_bootstrap as bootstrap_module
from aloy_backend.event_bootstrap import (
    EVENT_BOOTSTRAP_RUN_KIND,
    execute_claimed_event_bootstrap,
    queue_event_bootstrap_if_ready,
)
from aloy_backend.event_context import (
    EventBriefPayload,
    EventEvidenceRef,
    GroundedText,
    event_bootstrap_input_fingerprint,
    refresh_event_context_snapshot,
)
from aloy_backend.models import (
    Event,
    EventBrief,
    EventTrailEntry,
    KnowledgeEntry,
    Organization,
    OrganizationMembership,
    Run,
)


async def _seed_ready_event(session, *, event_id: str = "evt-bootstrap") -> None:
    session.add_all(
        [
            Organization(
                id="org-bootstrap",
                name="Bootstrap org",
                slug=f"bootstrap-{event_id}",
                created_by="alice",
                policy={},
            ),
            OrganizationMembership(
                organization_id="org-bootstrap",
                user_id="alice",
                role="member",
            ),
            Event(
                id=event_id,
                organization_id="org-bootstrap",
                user_id="alice",
                title="University",
                summary="Manage this semester and keep every course on track.",
            ),
            KnowledgeEntry(
                id=f"knowledge-{event_id}",
                organization_id="org-bootstrap",
                user_id="alice",
                event_id=event_id,
                content=(
                    "The semester includes algorithms, databases, exam dates, "
                    "coursework deadlines, study constraints, and weekly goals. "
                )
                * 12,
                source="user",
                provenance={"source": "event_setup"},
            ),
        ]
    )
    await session.flush()


async def test_queue_is_snapshot_bound_idempotent_and_freezes_evidence(
    db_session_maker,
):
    async with db_session_maker() as session:
        await _seed_ready_event(session)

        snapshot, run, created = await queue_event_bootstrap_if_ready(
            session,
            organization_id="org-bootstrap",
            user_id="alice",
            event_id="evt-bootstrap",
        )
        same_snapshot, same_run, created_again = await queue_event_bootstrap_if_ready(
            session,
            organization_id="org-bootstrap",
            user_id="alice",
            event_id="evt-bootstrap",
        )

        assert created is True
        assert created_again is False
        assert run is not None
        assert same_run is not None
        assert same_run.id == run.id
        assert same_snapshot.id == snapshot.id
        assert run.run_kind == EVENT_BOOTSTRAP_RUN_KIND
        assert run.context_snapshot_id == snapshot.id
        assert run.run_profile is not None
        assert run.run_profile["profile_id"] == "aloy.event-bootstrap"
        assert run.run_profile["version"] == "1"
        assert run.run_profile["fingerprint"]
        assert snapshot.evidence_payload[0]["id"] == "knowledge-evt-bootstrap"
        assert "algorithms" in snapshot.evidence_payload[0]["text"]
        assert (
            len(
                list(
                    (
                        await session.execute(
                            select(EventTrailEntry).where(
                                EventTrailEntry.kind == "event_bootstrap_queued"
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
            )
            == 1
        )


async def test_surface_lifecycle_activity_cannot_queue_another_bootstrap_run(
    db_session_maker,
):
    async with db_session_maker() as session:
        await _seed_ready_event(session)
        original_snapshot, original_run, created = await queue_event_bootstrap_if_ready(
            session,
            organization_id="org-bootstrap",
            user_id="alice",
            event_id="evt-bootstrap",
        )
        assert original_run is not None
        assert created is True
        session.add(
            EventTrailEntry(
                organization_id="org-bootstrap",
                user_id="alice",
                event_id="evt-bootstrap",
                actor_id="aloy:surface-builder",
                kind="surface_build_failed",
                summary="A rejected Surface candidate retained the last-good build",
                run_id="surface-run",
            )
        )
        await session.commit()

        (
            operational_snapshot,
            _pack,
            snapshot_created,
        ) = await refresh_event_context_snapshot(
            session,
            organization_id="org-bootstrap",
            user_id="alice",
            event_id="evt-bootstrap",
        )
        current_snapshot, same_run, queued = await queue_event_bootstrap_if_ready(
            session,
            organization_id="org-bootstrap",
            user_id="alice",
            event_id="evt-bootstrap",
        )

        assert snapshot_created is True
        assert operational_snapshot.id != original_snapshot.id
        assert current_snapshot.id == operational_snapshot.id
        assert event_bootstrap_input_fingerprint(
            operational_snapshot
        ) == event_bootstrap_input_fingerprint(original_snapshot)
        assert queued is False
        assert same_run is not None
        assert same_run.id == original_run.id
        runs = list(
            (
                await session.execute(
                    select(Run).where(Run.run_kind == EVENT_BOOTSTRAP_RUN_KIND)
                )
            )
            .scalars()
            .all()
        )
        assert len(runs) == 1


async def test_reading_event_workspace_never_queues_bootstrap_run(
    client,
    db_session_maker,
):
    async with db_session_maker() as session:
        await _seed_ready_event(session, event_id="evt-read-only")
        await session.commit()

    response = await client.get(
        "/v1/events/evt-read-only",
        headers={
            "X-Test-User": "alice",
            "X-Pori-Organization": "org-bootstrap",
        },
    )

    assert response.status_code == 200
    async with db_session_maker() as session:
        runs = list(
            (await session.execute(select(Run).where(Run.event_id == "evt-read-only")))
            .scalars()
            .all()
        )
        assert runs == []


async def test_bootstrap_executes_without_tools_and_publishes_grounded_brief(
    db_session_maker, monkeypatch
):
    monkeypatch.setattr(bootstrap_module, "async_session", db_session_maker)
    captured: dict = {}

    class FakeStructuredModel:
        async def ainvoke(self, messages):
            captured["messages"] = messages
            return {
                "parsed": EventBriefPayload(
                    purpose=GroundedText(
                        text="Keep the semester organized and on track.",
                        evidence_refs=[
                            EventEvidenceRef(
                                kind="knowledge_entry",
                                id="knowledge-evt-bootstrap",
                            )
                        ],
                    ),
                    unknowns=["The exact exam timetable is not known yet."],
                ),
                "raw": None,
            }

    class FakeModel:
        model = "fake-structured-model"

        def with_structured_output(self, output_model, *, include_raw=False):
            captured["output_model"] = output_model
            captured["include_raw"] = include_raw
            return FakeStructuredModel()

    def build_fake(**kwargs):
        captured["builder_kwargs"] = kwargs
        return SimpleNamespace(llm=FakeModel())

    async with db_session_maker() as session:
        await _seed_ready_event(session)
        snapshot, run, _ = await queue_event_bootstrap_if_ready(
            session,
            organization_id="org-bootstrap",
            user_id="alice",
            event_id="evt-bootstrap",
        )
        assert run is not None
        run.status = "running"
        run.attempt_count = 1
        run.lease_owner = "worker-a"
        run.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
        session.add_all(
            [
                run,
                EventTrailEntry(
                    organization_id="org-bootstrap",
                    user_id="alice",
                    event_id="evt-bootstrap",
                    actor_id="aloy:surface-builder",
                    kind="surface_candidate_rejected",
                    summary="A Surface candidate failed quality inspection",
                    run_id="surface-run",
                ),
            ]
        )
        await session.commit()
        run_id = run.id

    handled = await execute_claimed_event_bootstrap(
        run_id,
        "worker-a",
        orchestrator_builder=build_fake,
    )

    assert handled is True
    assert captured["builder_kwargs"]["allowed_tools"] == ()
    assert captured["builder_kwargs"]["allowed_capability_groups"] == ()
    assert captured["output_model"] is EventBriefPayload
    assert captured["include_raw"] is True
    prompt = captured["messages"][-1].content
    assert snapshot.id in prompt
    assert "algorithms" in prompt

    async with db_session_maker() as session:
        completed = await session.get(Run, run_id)
        brief = (await session.execute(select(EventBrief))).scalar_one()
        event = await session.get(Event, "evt-bootstrap")
        assert completed is not None
        assert completed.status == "completed"
        assert completed.success is True
        assert completed.lease_owner is None
        assert completed.metrics["model"] == "fake-structured-model"
        assert completed.metrics["structured_output"] is True
        assert completed.metrics["context_snapshot_version"] == 1
        assert completed.metrics["budget_usage"]["steps_used"] == 1
        assert completed.metrics["budget_usage"]["llm_calls_used"] == 1
        assert brief.creator_run_id == run_id
        assert brief.source_context_snapshot_id == snapshot.id
        assert event is not None
        assert event.metadata_["setup"]["bootstrap"]["brief_id"] == brief.id


async def test_stale_snapshot_is_cancelled_and_requeued_before_model_call(
    db_session_maker, monkeypatch
):
    monkeypatch.setattr(bootstrap_module, "async_session", db_session_maker)
    invoked = False

    def should_not_build(**kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("A stale snapshot must not reach the model")

    async with db_session_maker() as session:
        await _seed_ready_event(session)
        old_snapshot, run, _ = await queue_event_bootstrap_if_ready(
            session,
            organization_id="org-bootstrap",
            user_id="alice",
            event_id="evt-bootstrap",
        )
        assert run is not None
        run.status = "running"
        run.attempt_count = 1
        run.lease_owner = "worker-a"
        evidence = await session.get(KnowledgeEntry, "knowledge-evt-bootstrap")
        assert evidence is not None
        evidence.content += " The user added a newly confirmed midterm date."
        session.add_all([run, evidence])
        await session.commit()
        old_run_id = run.id

    assert await execute_claimed_event_bootstrap(
        old_run_id,
        "worker-a",
        orchestrator_builder=should_not_build,
    )
    assert invoked is False

    async with db_session_maker() as session:
        old_run = await session.get(Run, old_run_id)
        runs = list(
            (
                await session.execute(
                    select(Run).where(Run.run_kind == EVENT_BOOTSTRAP_RUN_KIND)
                )
            )
            .scalars()
            .all()
        )
        assert old_run is not None
        assert old_run.status == "cancelled"
        replacement = next(item for item in runs if item.id != old_run_id)
        assert replacement.status == "pending"
        assert replacement.context_snapshot_id != old_snapshot.id


async def test_invalid_model_evidence_fails_closed_and_can_be_retried(
    db_session_maker, monkeypatch
):
    monkeypatch.setattr(bootstrap_module, "async_session", db_session_maker)

    class InvalidStructuredModel:
        async def ainvoke(self, messages):
            return {
                "parsed": EventBriefPayload(
                    purpose=GroundedText(
                        text="Invented purpose",
                        evidence_refs=[
                            EventEvidenceRef(
                                kind="knowledge_entry", id="outside-snapshot"
                            )
                        ],
                    )
                ),
                "raw": None,
            }

    class InvalidModel:
        model = "fake-invalid-model"

        def with_structured_output(self, output_model, *, include_raw=False):
            return InvalidStructuredModel()

    async with db_session_maker() as session:
        await _seed_ready_event(session)
        _snapshot, run, _ = await queue_event_bootstrap_if_ready(
            session,
            organization_id="org-bootstrap",
            user_id="alice",
            event_id="evt-bootstrap",
        )
        assert run is not None
        run.status = "running"
        run.attempt_count = 1
        run.max_attempts = 1
        run.lease_owner = "worker-a"
        session.add(run)
        await session.commit()
        run_id = run.id

    assert await execute_claimed_event_bootstrap(
        run_id,
        "worker-a",
        orchestrator_builder=lambda **kwargs: SimpleNamespace(llm=InvalidModel()),
    )

    async with db_session_maker() as session:
        failed = await session.get(Run, run_id)
        assert failed is not None
        assert failed.status == "failed"
        assert failed.lease_owner is None
        assert (await session.execute(select(EventBrief))).scalars().first() is None


async def test_general_worker_dispatches_bootstrap_to_the_no_tool_executor(
    db_session_maker, monkeypatch
):
    monkeypatch.setattr(background_module, "async_session", db_session_maker)
    dispatched: list[tuple[str, str]] = []

    async def fake_executor(run_id: str, worker_id: str) -> bool:
        dispatched.append((run_id, worker_id))
        return True

    monkeypatch.setattr(
        background_module,
        "execute_claimed_event_bootstrap",
        fake_executor,
    )
    async with db_session_maker() as session:
        session.add(
            Event(
                id="evt-dispatch",
                organization_id="org-dispatch",
                user_id="alice",
                title="Dispatch",
            )
        )
        run = Run(
            organization_id="org-dispatch",
            user_id="alice",
            event_id="evt-dispatch",
            agent_id="aloy:event-bootstrap",
            session_id="evt-dispatch",
            run_kind=EVENT_BOOTSTRAP_RUN_KIND,
            task="Build the Event Brief",
            status="running",
            lease_owner="worker-a",
        )
        session.add(run)
        await session.commit()
        run_id = run.id

    await background_module.execute_claimed_run(run_id, "worker-a")

    assert dispatched == [(run_id, "worker-a")]


async def test_bootstrap_retry_endpoint_requeues_a_failed_run(client, db_session_maker):
    created = await client.post(
        "/v1/events",
        json={
            "title": "Career OS",
            "summary": "Research companies, roles, constraints, and application strategy. "
            * 12,
        },
    )
    event_id = created.json()["id"]
    async with db_session_maker() as session:
        run = (
            await session.execute(
                select(Run).where(
                    Run.event_id == event_id,
                    Run.run_kind == EVENT_BOOTSTRAP_RUN_KIND,
                )
            )
        ).scalar_one()
        run.status = "failed"
        run.attempt_count = run.max_attempts
        session.add(run)
        await session.commit()

    retried = await client.post(f"/v1/events/{event_id}/bootstrap")

    assert retried.status_code == 202
    assert retried.json()["status"] == "queued"
    assert retried.json()["attempt_count"] == 0
    assert retried.json()["can_retry"] is False


def test_event_bootstrap_migration_round_trip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'event-bootstrap.db'}")
    metadata = sa.MetaData()
    events = sa.Table(
        "events", metadata, sa.Column("id", sa.String(), primary_key=True)
    )
    sa.Table(
        "event_context_snapshots",
        metadata,
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("event_id", sa.String(), sa.ForeignKey(events.c.id)),
    )
    sa.Table(
        "runs",
        metadata,
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("event_id", sa.String(), sa.ForeignKey(events.c.id)),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        migration = importlib.import_module(
            "aloy_backend.alembic.versions.e7b8c9d0f1a2_event_bootstrap_runs"
        )
        original_op = migration.op
        migration.op = Operations(MigrationContext.configure(connection))
        try:
            migration.upgrade()
            snapshot_columns = {
                column["name"]
                for column in inspect(connection).get_columns("event_context_snapshots")
            }
            run_columns = {
                column["name"] for column in inspect(connection).get_columns("runs")
            }
            run_indexes = {
                index["name"] for index in inspect(connection).get_indexes("runs")
            }
            assert "evidence_payload" in snapshot_columns
            assert {"run_kind", "context_snapshot_id", "run_profile"} <= run_columns
            assert "uq_runs_event_context_kind" in run_indexes

            migration.downgrade()
            snapshot_columns = {
                column["name"]
                for column in inspect(connection).get_columns("event_context_snapshots")
            }
            run_columns = {
                column["name"] for column in inspect(connection).get_columns("runs")
            }
            assert "evidence_payload" not in snapshot_columns
            assert "run_kind" not in run_columns
        finally:
            migration.op = original_op
    engine.dispose()
