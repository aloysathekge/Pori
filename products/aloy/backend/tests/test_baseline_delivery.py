"""Baseline Surface delivery at Event creation (spec S3).

Creating a custom Event persists the reviewed baseline draft and queues the
model-free materialization Run; Life is exempt, existing Surface state is
never touched, replay is idempotent, and a delivery failure never blocks
Event creation. The baseline source's own gate proof lives in
``test_baseline_surface.py``; the worker-dispatch proof here uses the
injectable pipeline exactly like the other materialization tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import select

from aloy_backend import baseline_delivery as delivery_module
from aloy_backend.baseline_delivery import ensure_event_baseline_surface
from aloy_backend.baseline_surface import baseline_surface_fingerprint
from aloy_backend.config import settings
from aloy_backend.events import ensure_life_event
from aloy_backend.models import Event, Run, SurfaceProject, SurfaceRevision
from aloy_backend.surface_materialization import (
    SURFACE_MATERIALIZATION_RUN_KIND,
    execute_claimed_surface_materialization,
)
from aloy_backend.surface_pipeline import SurfacePipelineResult


@pytest.fixture(autouse=True)
def baseline_enabled():
    previous = settings.surface_baseline_enabled
    settings.surface_baseline_enabled = True
    yield
    settings.surface_baseline_enabled = previous


async def _created_event(client, db_session_maker, title="Baseline Event"):
    response = await client.post(
        "/v1/events",
        json={"title": title, "summary": "A fresh custom Event"},
    )
    assert response.status_code == 201
    return response.json()["id"]


async def test_event_creation_queues_the_baseline_materialization(
    client, db_session_maker
):
    event_id = await _created_event(client, db_session_maker)
    async with db_session_maker() as session:
        project = (
            (
                await session.execute(
                    select(SurfaceProject).where(SurfaceProject.event_id == event_id)
                )
            )
            .scalars()
            .one()
        )
        revision = await session.get(SurfaceRevision, project.draft_revision_id)
        run = (
            (
                await session.execute(
                    select(Run).where(
                        Run.event_id == event_id,
                        Run.run_kind == SURFACE_MATERIALIZATION_RUN_KIND,
                    )
                )
            )
            .scalars()
            .one()
        )
        assert revision is not None
        assert revision.request_fingerprint == baseline_surface_fingerprint()
        assert "/surface.json" in revision.files
        assert "/src/App.tsx" in revision.files
        assert run.status == "pending"
        assert run.model_assignment is None
        assert run.run_profile["revision_id"] == revision.id
        # Replay against the same Event changes nothing.
        event = await session.get(Event, event_id)
        assert (
            await ensure_event_baseline_surface(
                session, event=event, actor_id=event.user_id
            )
            == run.id
        )


async def test_setup_draft_promotion_also_delivers_the_baseline(
    client, db_session_maker
):
    created = await client.post(
        "/v1/event-drafts",
        json={"title": "Promoted Event", "description": "Planned via setup"},
    )
    assert created.status_code == 201
    draft_id = created.json()["id"]
    promoted = await client.post(f"/v1/event-drafts/{draft_id}/promote")
    assert promoted.status_code == 201
    event_id = promoted.json()["id"]
    async with db_session_maker() as session:
        run = (
            (
                await session.execute(
                    select(Run).where(
                        Run.event_id == event_id,
                        Run.run_kind == SURFACE_MATERIALIZATION_RUN_KIND,
                    )
                )
            )
            .scalars()
            .one()
        )
        assert run.status == "pending"
        assert run.model_assignment is None


async def test_life_and_events_with_existing_surface_state_are_exempt(
    client, db_session_maker
):
    event_id = await _created_event(client, db_session_maker)
    async with db_session_maker() as session:
        life = await ensure_life_event(
            session, organization_id="user:test-user", user_id="test-user"
        )
        assert (
            await ensure_event_baseline_surface(
                session, event=life, actor_id="test-user"
            )
            is None
        )
        # The created Event already owns baseline state: ensure() must not
        # create a second project.
        event = await session.get(Event, event_id)
        await ensure_event_baseline_surface(session, event=event, actor_id="test-user")
        await session.commit()
        projects = (
            (
                await session.execute(
                    select(SurfaceProject).where(SurfaceProject.event_id == event_id)
                )
            )
            .scalars()
            .all()
        )
        assert len(projects) == 1


async def test_delivery_failure_never_blocks_event_creation(
    client, db_session_maker, monkeypatch
):
    def broken_files():
        raise RuntimeError("bundled template unavailable")

    monkeypatch.setattr(delivery_module, "baseline_surface_files", broken_files)
    response = await client.post(
        "/v1/events",
        json={"title": "Still created", "summary": "Delivery failure tolerated"},
    )
    assert response.status_code == 201
    event_id = response.json()["id"]
    async with db_session_maker() as session:
        projects = (
            (
                await session.execute(
                    select(SurfaceProject).where(SurfaceProject.event_id == event_id)
                )
            )
            .scalars()
            .all()
        )
        assert projects == []


async def test_queued_baseline_run_drives_the_shared_pipeline_to_publication(
    client, db_session_maker, monkeypatch
):
    from aloy_backend import surface_materialization as materialization_module

    event_id = await _created_event(client, db_session_maker)
    async with db_session_maker() as session:
        run = (
            (
                await session.execute(
                    select(Run).where(
                        Run.event_id == event_id,
                        Run.run_kind == SURFACE_MATERIALIZATION_RUN_KIND,
                    )
                )
            )
            .scalars()
            .one()
        )
        run_id = run.id
        expected_revision = run.run_profile["revision_id"]
        run.status = "running"
        run.attempt_count = 1
        run.lease_owner = "worker-baseline"
        run.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
        run.started_at = datetime.now(timezone.utc)
        session.add(run)
        await session.commit()

    observed: dict = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            self.observe = kwargs["stage_observer"]

        async def execute(self, **kwargs):
            observed.update(kwargs)
            await self.observe("publishing_surface")
            return SurfacePipelineResult(
                status="published",
                candidate_fingerprint=kwargs["source_fingerprint"],
                revision_id=kwargs["revision_id"],
                build_id="build-baseline",
                publication={"id": "publication-baseline"},
            )

    async def verified(_session, *, run):
        return {
            "project_id": run.run_profile["project_id"],
            "publication_id": "publication-baseline",
            "revision_id": run.run_profile["revision_id"],
            "build_id": "build-baseline",
        }

    monkeypatch.setattr(
        materialization_module, "verified_surface_publication", verified
    )
    completed = await execute_claimed_surface_materialization(
        run_id,
        "worker-baseline",
        session_factory=db_session_maker,
        handler_factory=lambda **_kwargs: object(),
        pipeline_factory=FakePipeline,
    )
    assert completed is True
    assert observed["revision_id"] == expected_revision
    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        assert run is not None
        assert run.status == "completed"
        assert run.success is True
        assert run.progress["stage"] == "published"
