from __future__ import annotations

import importlib
from types import SimpleNamespace

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect
from sqlmodel import select

import aloy_backend.surface_requests as surface_requests_module
from aloy_backend.models import (
    Event,
    Run,
    SurfaceBuild,
    SurfaceEvolutionProposal,
    SurfaceProject,
    SurfaceRevision,
)
from aloy_backend.surface_evolution import SurfaceEvolutionSignal
from aloy_backend.surface_evolution_proposals import (
    decide_surface_evolution_proposal,
    record_surface_evolution_signal,
)
from aloy_backend.surface_interactions import (
    SurfaceInteractionError,
    SurfaceInteractionRequest,
    record_surface_interaction_rejection,
)
from aloy_backend.surface_manifest import SurfaceManifest
from aloy_backend.tenancy import OrganizationContext, OrganizationPolicy


async def _create_event(client) -> dict:
    response = await client.post(
        "/v1/events",
        json={"title": "University", "summary": "Semester", "phase": "active"},
    )
    assert response.status_code == 201
    return response.json()


def _context(event: Event) -> OrganizationContext:
    return OrganizationContext(
        organization_id=event.organization_id,
        user_id=event.user_id,
        role="member",
        permissions=(),
        policy=OrganizationPolicy(),
    )


async def _published_surface(session, event: Event) -> SurfaceProject:
    project = SurfaceProject(
        organization_id=event.organization_id,
        user_id=event.user_id,
        event_id=event.id,
        lifecycle="published",
    )
    session.add(project)
    await session.flush()
    revision = SurfaceRevision(
        organization_id=event.organization_id,
        user_id=event.user_id,
        event_id=event.id,
        project_id=project.id,
        revision_number=1,
        idempotency_key="evolution-revision-1",
        request_fingerprint="revision-fingerprint",
        manifest=SurfaceManifest().model_dump(mode="json"),
        files={"/src/App.tsx": "export default () => null"},
        checksum="source-checksum",
    )
    session.add(revision)
    await session.flush()
    build = SurfaceBuild(
        organization_id=event.organization_id,
        user_id=event.user_id,
        event_id=event.id,
        project_id=project.id,
        revision_id=revision.id,
        idempotency_key="evolution-build-1",
        request_fingerprint="build-fingerprint",
        status="succeeded",
        source_checksum=revision.checksum,
        toolchain_version="test",
        bundle_key="surfaces/test/bundle.zip",
        bundle_sha256="bundle-checksum",
        bundle_size_bytes=100,
    )
    session.add(build)
    await session.flush()
    project.published_revision_id = revision.id
    project.published_build_id = build.id
    session.add(project)
    await session.commit()
    return project


async def test_repeated_job_failure_aggregates_before_becoming_pending(
    client,
    db_session_maker,
):
    created = await _create_event(client)
    async with db_session_maker() as session:
        event = await session.get(Event, created["id"])
        assert event is not None
        await _published_surface(session, event)
        signal = SurfaceEvolutionSignal(
            trigger="primary_job_failure",
            goal="Adding an application fails",
            evidence_refs=["attempt-1"],
        )
        first = await record_surface_evolution_signal(
            session, context=_context(event), event=event, signal=signal
        )
        first_status = first.status
        second = await record_surface_evolution_signal(
            session,
            context=_context(event),
            event=event,
            signal=signal.model_copy(update={"evidence_refs": ["attempt-2"]}),
        )

    assert first_status == "observing"
    assert second.status == "pending"
    assert second.occurrence_count == 2
    assert second.evidence_refs == ["attempt-1", "attempt-2"]


async def test_user_feedback_endpoint_creates_pending_proposal(
    client,
    db_session_maker,
):
    created = await _create_event(client)
    async with db_session_maker() as session:
        event = await session.get(Event, created["id"])
        assert event is not None
        await _published_surface(session, event)

    response = await client.post(
        f"/v1/events/{created['id']}/surface/feedback",
        json={"message": "The timetable is difficult to use on my phone"},
    )

    assert response.status_code == 200
    assert response.json()["trigger"] == "negative_feedback"
    assert response.json()["status"] == "pending"


async def test_repeated_rejected_surface_command_creates_one_proposal(
    client,
    db_session_maker,
):
    created = await _create_event(client)
    async with db_session_maker() as session:
        event = await session.get(Event, created["id"])
        assert event is not None
        project = await _published_surface(session, event)
        assert project.published_build_id is not None
        assert project.published_revision_id is not None
        context = _context(event)
        for index in range(2):
            error = SurfaceInteractionError(
                422,
                "The save payload is invalid",
                code="invalid_payload",
            )
            attempt = await record_surface_interaction_rejection(
                session,
                context=context,
                event_id=event.id,
                request=SurfaceInteractionRequest(
                    build_id=project.published_build_id,
                    code_revision_id=project.published_revision_id,
                    data_revision=0,
                    method="command",
                    name="career.application_save",
                    component_id="application-form",
                    payload={"company": "Example"},
                    idempotency_key=f"failed-save-{index}",
                ),
                error=error,
            )
            assert attempt is not None

        proposals = list(
            (
                await session.execute(
                    select(SurfaceEvolutionProposal).where(
                        SurfaceEvolutionProposal.event_id == event.id,
                        SurfaceEvolutionProposal.trigger == "primary_job_failure",
                    )
                )
            )
            .scalars()
            .all()
        )

    assert len(proposals) == 1
    assert proposals[0].status == "pending"
    assert proposals[0].occurrence_count == 2
    assert len(proposals[0].evidence_refs) == 2


async def test_event_phase_change_creates_surface_proposal(
    client,
    db_session_maker,
):
    created = await _create_event(client)
    async with db_session_maker() as session:
        event = await session.get(Event, created["id"])
        assert event is not None
        await _published_surface(session, event)

    response = await client.patch(
        f"/v1/events/{created['id']}",
        json={"phase": "Exam preparation"},
    )

    assert response.status_code == 200
    assert response.json()["phase"] == "Exam preparation"
    proposals = await client.get(
        f"/v1/events/{created['id']}/surface/evolution-proposals"
    )
    assert proposals.status_code == 200
    assert proposals.json()[0]["trigger"] == "event_phase_changed"
    assert "Exam preparation" in proposals.json()[0]["goal"]


async def test_dismissal_creates_cooldown_and_hides_pending_proposal(
    client,
    db_session_maker,
):
    created = await _create_event(client)
    async with db_session_maker() as session:
        event = await session.get(Event, created["id"])
        assert event is not None
        await _published_surface(session, event)
        proposal = await record_surface_evolution_signal(
            session,
            context=_context(event),
            event=event,
            signal=SurfaceEvolutionSignal(
                trigger="negative_feedback",
                goal="The timetable is not useful",
            ),
        )
        dismissed = await decide_surface_evolution_proposal(
            session,
            context=_context(event),
            event=event,
            proposal_id=proposal.id,
            decision="dismiss",
        )

    assert dismissed.status == "dismissed"
    assert dismissed.cooldown_until is not None
    response = await client.get(
        f"/v1/events/{created['id']}/surface/evolution-proposals"
    )
    assert response.status_code == 200
    assert response.json() == []


async def test_acceptance_queues_exactly_one_builder_run(
    client,
    db_session_maker,
    monkeypatch,
):
    created = await _create_event(client)
    assignment = SimpleNamespace(
        role=SimpleNamespace(value="surface_builder"),
        provider="test-provider",
        model="test-builder",
        skill_id="surface-builder@1",
        config_fingerprint="builder-config",
        descriptor=lambda: {
            "role": "surface_builder",
            "provider": "test-provider",
            "model": "test-builder",
        },
    )
    monkeypatch.setattr(
        surface_requests_module,
        "resolve_model_assignment",
        lambda *_args, **_kwargs: assignment,
    )
    async with db_session_maker() as session:
        event = await session.get(Event, created["id"])
        assert event is not None
        await _published_surface(session, event)
        proposal = await record_surface_evolution_signal(
            session,
            context=_context(event),
            event=event,
            signal=SurfaceEvolutionSignal(
                trigger="capability_gap",
                goal="Add a map for campus locations",
            ),
        )
        accepted = await decide_surface_evolution_proposal(
            session,
            context=_context(event),
            event=event,
            proposal_id=proposal.id,
            decision="accept",
        )
        runs = list(
            (await session.execute(select(Run).where(Run.event_id == event.id)))
            .scalars()
            .all()
        )
        persisted = await session.get(SurfaceEvolutionProposal, proposal.id)

    assert accepted.status == "queued"
    assert persisted is not None
    assert len(runs) == 1
    assert runs[0].id == accepted.builder_run_id
    assert any(
        receipt.get("kind") == "surface_evolution_decision"
        for receipt in runs[0].execution_receipts or []
    )


def test_surface_evolution_proposal_migration_round_trip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'surface-evolution.db'}")
    metadata = sa.MetaData()
    for table_name in (
        "events",
        "surface_projects",
        "surface_revisions",
        "surface_builds",
        "runs",
    ):
        sa.Table(table_name, metadata, sa.Column("id", sa.String(), primary_key=True))
    metadata.create_all(engine)

    with engine.begin() as connection:
        migration = importlib.import_module(
            "aloy_backend.alembic.versions." "o7f8a9b0c1d2_surface_evolution_proposals"
        )
        original_op = migration.op
        migration.op = Operations(MigrationContext.configure(connection))
        try:
            migration.upgrade()
            columns = {
                column["name"]
                for column in inspect(connection).get_columns(
                    "surface_evolution_proposals"
                )
            }
            assert {
                "signal_fingerprint",
                "occurrence_count",
                "cooldown_until",
                "builder_run_id",
            } <= columns

            migration.downgrade()
            assert (
                "surface_evolution_proposals"
                not in inspect(connection).get_table_names()
            )
        finally:
            migration.op = original_op
