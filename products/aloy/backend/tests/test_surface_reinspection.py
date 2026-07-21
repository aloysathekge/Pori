from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from sqlmodel import select

from aloy_backend import background as background_module
from aloy_backend import surface_requests as surface_requests_module
from aloy_backend.models import (
    Event,
    Run,
    SurfaceBuild,
    SurfaceEvolutionProposal,
    SurfaceProject,
    SurfaceRevision,
)
from aloy_backend.surface_reinspection import (
    SURFACE_REINSPECTION_RUN_KIND,
    execute_claimed_surface_reinspection,
    queue_due_surface_reinspections,
)


async def _create_published_surface(
    client,
    db_session_maker,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[dict, str]:
    response = await client.post(
        "/v1/events",
        headers=headers,
        json={"title": "University", "summary": "Semester", "phase": "active"},
    )
    assert response.status_code == 201
    created = response.json()
    async with db_session_maker() as session:
        event = await session.get(Event, created["id"])
        assert event is not None
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
            idempotency_key="reinspection-revision",
            request_fingerprint="revision-fingerprint",
            manifest={},
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
            idempotency_key="reinspection-build",
            request_fingerprint="build-fingerprint",
            status="succeeded",
            source_checksum=revision.checksum,
            toolchain_version="test",
            bundle_key="surfaces/test/bundle.zip",
            bundle_sha256="bundle-checksum",
            bundle_size_bytes=100,
            resource_metrics={
                "surface_quality": {
                    "passed": True,
                    "fingerprint": "publication-quality",
                }
            },
        )
        session.add(build)
        await session.flush()
        project.published_revision_id = revision.id
        project.published_build_id = build.id
        session.add(project)
        await session.commit()
        return created, build.id


async def _claim_run(db_session_maker, run_id: str, worker_id: str) -> None:
    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        assert run is not None
        run.status = "running"
        run.lease_owner = worker_id
        run.attempt_count = 1
        run.started_at = datetime.now(timezone.utc)
        session.add(run)
        await session.commit()


class _FakeInspectionHandler:
    def __init__(self, *, result: dict, **_kwargs):
        self.result = result

    async def preview(self, params):
        assert params.force_reinspection is True
        assert params.inspection_kind == "reinspection"
        return self.result


def _handler_factory(result: dict):
    return lambda **kwargs: _FakeInspectionHandler(result=result, **kwargs)


async def test_reinspection_endpoint_queues_and_deduplicates_active_run(
    client,
    db_session_maker,
):
    created, build_id = await _create_published_surface(client, db_session_maker)

    first = await client.post(
        f"/v1/events/{created['id']}/surface/operator/reinspections",
        json={"reason": "daily_health_check"},
    )
    second = await client.post(
        f"/v1/events/{created['id']}/surface/operator/reinspections",
        json={"reason": "runtime_change"},
    )

    assert first.status_code == 202
    assert first.json()["build_id"] == build_id
    assert first.json()["replayed"] is False
    assert second.status_code == 202
    assert second.json()["run_id"] == first.json()["run_id"]
    assert second.json()["replayed"] is True

    health = await client.get(f"/v1/events/{created['id']}/surface/operator/health")
    assert health.status_code == 200
    assert health.json()["status"] == "checking"
    assert health.json()["build_id"] == build_id
    assert health.json()["run_id"] == first.json()["run_id"]


async def test_surface_health_controls_require_operator_authority(
    client,
    db_session_maker,
):
    organization = await client.post(
        "/v1/organizations",
        headers={"X-Test-User": "alice"},
        json={"name": "Internal boundary", "slug": "internal-boundary"},
    )
    assert organization.status_code == 201
    organization_id = organization.json()["id"]
    owner_headers = {
        "X-Test-User": "alice",
        "X-Pori-Organization": organization_id,
    }
    added = await client.post(
        f"/v1/organizations/{organization_id}/members",
        headers=owner_headers,
        json={"user_id": "bob", "role": "member"},
    )
    assert added.status_code == 201
    member_headers = {
        "X-Test-User": "bob",
        "X-Pori-Organization": organization_id,
    }
    created, _ = await _create_published_surface(
        client,
        db_session_maker,
        headers=member_headers,
    )

    health = await client.get(
        f"/v1/events/{created['id']}/surface/operator/health",
        headers=member_headers,
    )
    reinspection = await client.post(
        f"/v1/events/{created['id']}/surface/operator/reinspections",
        headers=member_headers,
        json={"reason": "manual_health_check"},
    )

    assert health.status_code == 403
    assert reinspection.status_code == 403


async def test_failed_trusted_reinspection_proposes_evolution(
    client,
    db_session_maker,
    monkeypatch,
):
    assignment = SimpleNamespace(
        role=SimpleNamespace(value="surface_builder"),
        provider="test-provider",
        model="fixed-test-builder",
        skill_id="surface-builder@1",
        config_fingerprint="fixed-test-builder-config",
        descriptor=lambda: {
            "role": "surface_builder",
            "provider": "test-provider",
            "model": "fixed-test-builder",
        },
    )
    monkeypatch.setattr(
        surface_requests_module,
        "resolve_model_assignment",
        lambda *_args, **_kwargs: assignment,
    )
    created, build_id = await _create_published_surface(client, db_session_maker)
    queued = await client.post(
        f"/v1/events/{created['id']}/surface/operator/reinspections",
        json={"reason": "runtime_change"},
    )
    run_id = queued.json()["run_id"]
    await _claim_run(db_session_maker, run_id, "worker-1")

    completed = await execute_claimed_surface_reinspection(
        run_id,
        "worker-1",
        session_factory=db_session_maker,
        handler_factory=_handler_factory(
            {
                "inspection_id": "sinspect-regression",
                "quality_gate": {
                    "passed": False,
                    "fingerprint": "failed-quality",
                },
            }
        ),
    )

    assert completed is True
    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        assert run is not None
        assert run.status == "completed"
        assert run.success is True
        assert run.progress["stage"] == "quality_regression"
        proposal = (
            (
                await session.execute(
                    select(SurfaceEvolutionProposal).where(
                        SurfaceEvolutionProposal.event_id == created["id"],
                        SurfaceEvolutionProposal.trigger == "quality_failure",
                    )
                )
            )
            .scalars()
            .first()
        )
        assert proposal is not None
        assert proposal.status == "pending"
        assert proposal.base_build_id == build_id
        assert proposal.evidence_refs == ["sinspect-regression"]

    health = await client.get(f"/v1/events/{created['id']}/surface/operator/health")
    assert health.status_code == 200
    assert health.json()["status"] == "needs_improvement"
    assert health.json()["build_id"] == build_id

    pending = await client.get(
        f"/v1/events/{created['id']}/surface/evolution-proposals"
    )
    proposal_id = pending.json()[0]["id"]
    accepted = await client.post(
        f"/v1/events/{created['id']}/surface/evolution-proposals/{proposal_id}/decision",
        json={"decision": "accept"},
    )
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "queued"
    assert accepted.json()["builder_run_id"]

    runtime = await client.get(f"/v1/events/{created['id']}/surface/runtime")
    assert runtime.status_code == 200
    assert runtime.json()["published_build_id"] == build_id


async def test_inspector_outage_retries_without_quality_proposal(
    client,
    db_session_maker,
):
    created, _ = await _create_published_surface(client, db_session_maker)
    queued = await client.post(
        f"/v1/events/{created['id']}/surface/operator/reinspections",
        json={"reason": "daily_health_check"},
    )
    run_id = queued.json()["run_id"]
    await _claim_run(db_session_maker, run_id, "worker-2")

    class UnavailableHandler:
        def __init__(self, **_kwargs):
            pass

        async def preview(self, _params):
            raise RuntimeError("inspector unavailable")

    completed = await execute_claimed_surface_reinspection(
        run_id,
        "worker-2",
        session_factory=db_session_maker,
        handler_factory=UnavailableHandler,
    )

    assert completed is False
    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        assert run is not None
        assert run.status == "pending"
        assert run.progress["stage"] == "retry_scheduled"
        proposals = list(
            (
                await session.execute(
                    select(SurfaceEvolutionProposal).where(
                        SurfaceEvolutionProposal.event_id == created["id"]
                    )
                )
            )
            .scalars()
            .all()
        )
        assert proposals == []


async def test_passing_reinspection_completes_without_proposal(
    client,
    db_session_maker,
):
    created, _ = await _create_published_surface(client, db_session_maker)
    queued = await client.post(
        f"/v1/events/{created['id']}/surface/operator/reinspections",
        json={"reason": "runtime_change"},
    )
    run_id = queued.json()["run_id"]
    await _claim_run(db_session_maker, run_id, "worker-3")

    completed = await execute_claimed_surface_reinspection(
        run_id,
        "worker-3",
        session_factory=db_session_maker,
        handler_factory=_handler_factory(
            {
                "inspection_id": "sinspect-passed",
                "quality_gate": {
                    "passed": True,
                    "fingerprint": "passed-quality",
                },
            }
        ),
    )

    assert completed is True
    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        assert run is not None
        assert run.progress["stage"] == "passed"
        proposal = (
            (
                await session.execute(
                    select(SurfaceEvolutionProposal).where(
                        SurfaceEvolutionProposal.event_id == created["id"]
                    )
                )
            )
            .scalars()
            .first()
        )
        assert proposal is None

    health = await client.get(f"/v1/events/{created['id']}/surface/operator/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ready"
    assert health.json()["inspection_id"] == "sinspect-passed"


async def test_surface_health_distinguishes_inspector_outage_from_regression(
    client,
    db_session_maker,
):
    created, build_id = await _create_published_surface(client, db_session_maker)
    queued = await client.post(
        f"/v1/events/{created['id']}/surface/operator/reinspections",
        json={"reason": "manual_health_check"},
    )
    run_id = queued.json()["run_id"]
    await _claim_run(db_session_maker, run_id, "worker-outage")
    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        assert run is not None
        run.attempt_count = run.max_attempts
        session.add(run)
        await session.commit()

    class UnavailableHandler:
        def __init__(self, **_kwargs):
            pass

        async def preview(self, _params):
            raise RuntimeError("inspector unavailable")

    completed = await execute_claimed_surface_reinspection(
        run_id,
        "worker-outage",
        session_factory=db_session_maker,
        handler_factory=UnavailableHandler,
    )
    assert completed is False

    health = await client.get(f"/v1/events/{created['id']}/surface/operator/health")
    assert health.status_code == 200
    assert health.json()["status"] == "unavailable"
    assert health.json()["build_id"] == build_id
    proposals = await client.get(
        f"/v1/events/{created['id']}/surface/evolution-proposals"
    )
    assert proposals.json() == []


async def test_worker_dispatches_model_free_reinspection_run(
    client,
    db_session_maker,
    monkeypatch,
):
    created, _ = await _create_published_surface(client, db_session_maker)
    queued = await client.post(
        f"/v1/events/{created['id']}/surface/operator/reinspections",
        json={"reason": "runtime_change"},
    )
    run_id = queued.json()["run_id"]
    await _claim_run(db_session_maker, run_id, "worker-dispatch")
    captured: dict[str, str] = {}

    async def fake_reinspection(received_run_id: str, worker_id: str) -> bool:
        captured.update({"run_id": received_run_id, "worker_id": worker_id})
        return True

    monkeypatch.setattr(background_module, "async_session", db_session_maker)
    monkeypatch.setattr(
        background_module,
        "execute_claimed_surface_reinspection",
        fake_reinspection,
    )

    await background_module.execute_claimed_run(run_id, "worker-dispatch")

    assert captured == {"run_id": run_id, "worker_id": "worker-dispatch"}


async def test_due_planner_throttles_checks_per_live_build(
    client,
    db_session_maker,
):
    created, _ = await _create_published_surface(client, db_session_maker)
    moment = datetime.now(timezone.utc)

    first = await queue_due_surface_reinspections(
        now=moment,
        interval_seconds=86_400,
        session_factory=db_session_maker,
    )
    immediate = await queue_due_surface_reinspections(
        now=moment,
        interval_seconds=86_400,
        session_factory=db_session_maker,
    )

    assert first == 1
    assert immediate == 0
    async with db_session_maker() as session:
        run = (
            (
                await session.execute(
                    select(Run).where(
                        Run.event_id == created["id"],
                        Run.run_kind == SURFACE_REINSPECTION_RUN_KIND,
                    )
                )
            )
            .scalars()
            .first()
        )
        assert run is not None
        assert run.run_profile["reason"] == "automatic_health_check"
        run.status = "completed"
        run.success = True
        run.completed_at = moment
        session.add(run)
        await session.commit()

    after_interval = await queue_due_surface_reinspections(
        now=moment + timedelta(days=2),
        interval_seconds=86_400,
        session_factory=db_session_maker,
    )
    assert after_interval == 1


def test_reinspection_run_kind_is_model_free():
    assert SURFACE_REINSPECTION_RUN_KIND == "surface_reinspection"
