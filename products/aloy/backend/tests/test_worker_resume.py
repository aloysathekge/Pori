"""Phase 2 resume-not-restart tests: re-claimed runs continue from their
persisted checkpoint, per-step checkpoints renew the lease, and salvage
partial results surface on the Run row (docs/long-running.md)."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from aloy_backend.background import (
    _make_progress_checkpointer,
    execute_claimed_run,
    kernel_task_id_for_run,
)
from aloy_backend.models import Organization, OrganizationMembership, Run

pytestmark = pytest.mark.asyncio


async def _seed_org_and_run(db_session_maker, run_kwargs=None):
    async with db_session_maker() as session:
        session.add(
            Organization(
                id="org-1",
                name="Org",
                slug="resume-org",
                created_by="alice",
                policy={},
            )
        )
        session.add(
            OrganizationMembership(
                organization_id="org-1",
                user_id="alice",
                role="member",
            )
        )
        run = Run(
            organization_id="org-1",
            user_id="alice",
            event_id="evt-worker-resume",
            agent_id="agent-1",
            session_id="session-1",
            task="long task",
            status="running",
            attempt_count=1,
            lease_owner="worker-a",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            **(run_kwargs or {}),
        )
        session.add(run)
        await session.commit()
        return run.id


async def test_reclaim_resumes_kernel_task_from_checkpoint(
    db_session_maker, monkeypatch
):
    monkeypatch.setattr("aloy_backend.background.async_session", db_session_maker)
    captured = {}

    class FakeOrchestrator:
        async def execute_task(self, **kwargs):
            captured.update(kwargs)
            return {
                "success": True,
                "steps_taken": 5,
                "agent": None,
                "result": {"metrics": None},
                "trace": {},
            }

    monkeypatch.setattr(
        "aloy_backend.background.build_orchestrator",
        lambda **kwargs: FakeOrchestrator(),
    )

    run_id = None
    async with db_session_maker() as session:
        pass  # ensure schema exists via fixture
    run_id = await _seed_org_and_run(db_session_maker)
    kernel_task_id = kernel_task_id_for_run(run_id)
    # Simulate a prior attempt that checkpointed 3 steps before dying
    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        run.attempt_count = 2
        run.progress = {
            "kernel_task_id": kernel_task_id,
            "n_steps": 3,
            "consecutive_failures": 1,
            "current_activity": "halfway through",
            "plan": [{"id": "1", "content": "keep going", "status": "in_progress"}],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        session.add(run)
        await session.commit()

    await execute_claimed_run(run_id, "worker-a")

    # The orchestrator was asked to RESUME the same kernel task…
    assert captured["resume_task_id"] == kernel_task_id
    assert callable(captured["on_step_end"])
    # …and the injected memory carries the checkpoint it will resume from.
    memory = captured["memory"]
    record = memory.tasks[kernel_task_id]
    assert record.n_steps == 3
    assert record.plan[0]["content"] == "keep going"
    # A fresh attempt does not inherit the failure streak
    assert record.consecutive_failures == 0


async def test_first_attempt_gets_stable_task_id_without_checkpoint(
    db_session_maker, monkeypatch
):
    monkeypatch.setattr("aloy_backend.background.async_session", db_session_maker)
    captured = {}

    class FakeOrchestrator:
        async def execute_task(self, **kwargs):
            captured.update(kwargs)
            return {
                "success": True,
                "steps_taken": 1,
                "agent": None,
                "result": {"metrics": None},
                "trace": {},
            }

    monkeypatch.setattr(
        "aloy_backend.background.build_orchestrator",
        lambda **kwargs: FakeOrchestrator(),
    )
    run_id = await _seed_org_and_run(db_session_maker)

    await execute_claimed_run(run_id, "worker-a")

    assert captured["resume_task_id"] == kernel_task_id_for_run(run_id)
    # No prior progress -> no injected task record; the Agent creates it fresh
    assert kernel_task_id_for_run(run_id) not in captured["memory"].tasks


async def test_step_checkpoint_persists_progress_and_renews_lease(
    db_session_maker, monkeypatch
):
    monkeypatch.setattr("aloy_backend.background.async_session", db_session_maker)
    run_id = await _seed_org_and_run(db_session_maker)
    kernel_task_id = kernel_task_id_for_run(run_id)

    # Shrink the lease so renewal is observable
    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        run.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=1)
        session.add(run)
        await session.commit()

    checkpoint = _make_progress_checkpointer(run_id, "worker-a", kernel_task_id)
    fake_agent = SimpleNamespace(
        state=SimpleNamespace(
            n_steps=7, consecutive_failures=0, current_activity="crunching"
        ),
        _plan_snapshot=lambda: [
            {"id": "1", "content": "step seven", "status": "in_progress"}
        ],
    )
    await checkpoint(fake_agent)

    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        assert run.progress["n_steps"] == 7
        assert run.progress["kernel_task_id"] == kernel_task_id
        assert run.progress["current_activity"] == "crunching"
        assert run.steps_taken == 7
        # Heartbeat: the lease was pushed well past the shrunken expiry
        # (SQLite loses tzinfo on round-trip, so normalize before comparing)
        expires = run.lease_expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        assert expires > datetime.now(timezone.utc) + timedelta(seconds=60)


async def test_checkpoint_refuses_when_lease_lost(db_session_maker, monkeypatch):
    monkeypatch.setattr("aloy_backend.background.async_session", db_session_maker)
    run_id = await _seed_org_and_run(db_session_maker)

    checkpoint = _make_progress_checkpointer(
        run_id, "some-other-worker", kernel_task_id_for_run(run_id)
    )
    fake_agent = SimpleNamespace(
        state=SimpleNamespace(
            n_steps=9, consecutive_failures=0, current_activity="zombie"
        ),
        _plan_snapshot=lambda: [],
    )
    await checkpoint(fake_agent)

    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        # The zombie worker (lease lost) must not overwrite progress
        assert run.progress is None


async def test_salvage_partial_result_surfaces_on_run(db_session_maker, monkeypatch):
    monkeypatch.setattr("aloy_backend.background.async_session", db_session_maker)

    class FakeOrchestrator:
        async def execute_task(self, **kwargs):
            return {
                "success": False,
                "steps_taken": 15,
                "agent": None,
                "result": {
                    "metrics": None,
                    "partial_result": {
                        "summary": "Got through most of the analysis.",
                        "reason": "step_limit",
                        "steps_taken": 15,
                    },
                },
                "trace": {},
            }

    monkeypatch.setattr(
        "aloy_backend.background.build_orchestrator",
        lambda **kwargs: FakeOrchestrator(),
    )
    run_id = await _seed_org_and_run(db_session_maker)

    await execute_claimed_run(run_id, "worker-a")

    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        assert run.status == "completed"
        assert run.success is False
        assert run.final_answer == "Got through most of the analysis."
        assert "step_limit" in (run.reasoning or "")
