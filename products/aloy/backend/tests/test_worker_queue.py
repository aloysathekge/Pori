from datetime import datetime, timedelta, timezone

import pytest
from pydantic import BaseModel
from sqlmodel import select

from aloy_backend.background import execute_claimed_run
from aloy_backend.models import (
    ActionProposal,
    Event,
    EventTrailEntry,
    Organization,
    OrganizationMembership,
    Run,
    TraceRecord,
)
from aloy_backend.worker import claim_next_run
from pori import ActionRequest, ApprovalRequest, ReviewConfig
from pori.tools.registry import ToolRegistry

pytestmark = pytest.mark.asyncio


async def test_api_enqueues_without_executing(client):
    created = await client.post(
        "/v1/runs",
        json={"task": "queued work", "max_steps": 3},
    )

    assert created.status_code == 202
    body = created.json()
    assert body["status"] == "pending"
    assert body["attempt_count"] == 0
    assert body["lease_owner"] is None
    assert body["isolation_profile"] == "worker-process"

    cancelled = await client.post(f"/v1/runs/{body['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["cancel_requested"] is True


async def test_worker_claims_pending_and_recovers_expired_lease(
    db_session_maker, monkeypatch
):
    monkeypatch.setattr("aloy_backend.worker.async_session", db_session_maker)
    async with db_session_maker() as session:
        pending = Run(
            organization_id="org-1",
            user_id="alice",
            event_id="evt-worker",
            agent_id="agent-1",
            session_id="session-1",
            task="pending",
        )
        expired = Run(
            organization_id="org-1",
            user_id="alice",
            event_id="evt-worker",
            agent_id="agent-1",
            session_id="session-2",
            task="expired",
            status="running",
            lease_owner="dead-worker",
            lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=5),
        )
        cancelled = Run(
            organization_id="org-1",
            user_id="alice",
            event_id="evt-worker",
            agent_id="agent-1",
            session_id="session-3",
            task="cancelled",
            cancel_requested=True,
        )
        session.add_all([pending, expired, cancelled])
        await session.commit()
        pending_id = pending.id
        expired_id = expired.id

    first_id = await claim_next_run("worker-a")
    second_id = await claim_next_run("worker-b")
    third_id = await claim_next_run("worker-c")

    assert {first_id, second_id} == {pending_id, expired_id}
    assert third_id is None
    async with db_session_maker() as session:
        claimed = await session.get(Run, first_id)
        assert claimed.status == "running"
        assert claimed.lease_owner in {"worker-a", "worker-b"}
        assert claimed.attempt_count == 1


async def test_worker_enforces_org_wide_concurrency_and_reclamps_budget(
    db_session_maker, monkeypatch
):
    monkeypatch.setattr("aloy_backend.worker.async_session", db_session_maker)
    now = datetime.now(timezone.utc)
    async with db_session_maker() as session:
        session.add(
            Organization(
                id="org-budget-cap",
                name="Budget cap",
                slug="budget-cap",
                created_by="alice",
                policy={
                    "max_concurrent_runs": 1,
                    "max_steps_per_run": 4,
                    "max_tool_calls_per_run": 5,
                    "max_tokens_per_run": 1_000,
                    "max_cost_usd_per_run": 0.5,
                    "run_timeout_seconds": 60,
                },
            )
        )
        active = Run(
            organization_id="org-budget-cap",
            user_id="alice",
            event_id="evt-active",
            agent_id="agent-1",
            session_id="session-active",
            task="already running",
            status="running",
            lease_owner="worker-active",
            lease_expires_at=now + timedelta(minutes=5),
        )
        pending = Run(
            organization_id="org-budget-cap",
            user_id="bob",
            event_id="evt-pending",
            agent_id="agent-2",
            session_id="session-pending",
            task="must wait",
            max_steps=99,
            max_tool_calls=99,
            max_tokens=99_999,
            max_cost_usd=9.0,
            timeout_seconds=999,
        )
        session.add_all([active, pending])
        await session.commit()
        active_id = active.id
        pending_id = pending.id

    assert await claim_next_run("worker-b") is None

    async with db_session_maker() as session:
        active = await session.get(Run, active_id)
        assert active is not None
        active.status = "completed"
        active.lease_owner = None
        active.lease_expires_at = None
        session.add(active)
        await session.commit()

    assert await claim_next_run("worker-b") == pending_id
    async with db_session_maker() as session:
        claimed = await session.get(Run, pending_id)
    assert claimed is not None
    assert claimed.max_steps == 4
    assert claimed.max_tool_calls == 5
    assert claimed.max_tokens == 1_000
    assert claimed.max_cost_usd == 0.5
    assert claimed.timeout_seconds == 60


async def test_worker_persists_budget_exhaustion_as_terminal_outcome(
    db_session_maker, monkeypatch
):
    monkeypatch.setattr("aloy_backend.background.async_session", db_session_maker)

    class FakeOrchestrator:
        async def execute_task(self, **kwargs):
            return {
                "success": False,
                "steps_taken": 2,
                "agent": None,
                "result": {
                    "completed": False,
                    "metrics": None,
                    "stop_reason": "max_tool_calls",
                    "budget_error": "Tool-call budget exceeded",
                    "budget_usage": {
                        "max_tool_calls": 2,
                        "tool_calls_used": 2,
                        "steps_used": 2,
                        "duration_seconds_used": 1.25,
                    },
                },
                "trace": {},
            }

    monkeypatch.setattr(
        "aloy_backend.background.build_orchestrator",
        lambda **kwargs: FakeOrchestrator(),
    )
    async with db_session_maker() as session:
        session.add(
            Organization(
                id="org-budget-outcome",
                name="Budget outcome",
                slug="budget-outcome",
                created_by="alice",
                policy={},
            )
        )
        session.add(
            OrganizationMembership(
                organization_id="org-budget-outcome",
                user_id="alice",
                role="member",
            )
        )
        session.add(
            Event(
                id="evt-budget-outcome",
                organization_id="org-budget-outcome",
                user_id="alice",
                title="Budget outcome",
            )
        )
        run = Run(
            organization_id="org-budget-outcome",
            user_id="alice",
            event_id="evt-budget-outcome",
            agent_id="agent-1",
            session_id="session-1",
            task="bounded work",
            status="running",
            attempt_count=1,
            lease_owner="worker-a",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            progress={
                "budget_attempt_started_at": datetime.now(timezone.utc).isoformat()
            },
        )
        session.add(run)
        await session.commit()
        run_id = run.id

    await execute_claimed_run(run_id, "worker-a")

    async with db_session_maker() as session:
        completed = await session.get(Run, run_id)
        trail = (
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.run_id == run_id,
                        EventTrailEntry.kind == "run_budget_exhausted",
                    )
                )
            )
            .scalars()
            .all()
        )

    assert completed is not None
    assert completed.status == "completed"
    assert completed.success is False
    assert completed.metrics["budget_gate"]["reason"] == "max_tool_calls"
    assert completed.progress["budget_usage"]["tool_calls_used"] == 2
    assert completed.execution_receipts[-1]["kind"] == "run_budget"
    assert len(trail) == 1


async def test_leased_worker_revalidates_membership_and_persists_trace(
    db_session_maker, monkeypatch
):
    monkeypatch.setattr("aloy_backend.background.async_session", db_session_maker)

    class FakeOrchestrator:
        async def execute_task(self, **kwargs):
            return {
                "success": True,
                "steps_taken": 1,
                "agent": None,
                "selected_skills": ["teach@1"],
                "artifacts": [
                    {
                        "kind": "file",
                        "tool_name": "write_file",
                        "path": "lessons/division.html",
                    }
                ],
                "plan": [
                    {
                        "id": "1",
                        "content": "answer the user",
                        "status": "completed",
                    }
                ],
                "result": {"metrics": None},
                "trace": {
                    "duration": "0.010s",
                    "total_spans": 1,
                    "status": "ok",
                    "prompt_fingerprint": "prompt-fp",
                    "tool_surface_fingerprint": "tools-fp",
                    "execution_receipts": [],
                },
            }

    monkeypatch.setattr(
        "aloy_backend.background.build_orchestrator",
        lambda **kwargs: FakeOrchestrator(),
    )
    async with db_session_maker() as session:
        session.add(
            Organization(
                id="org-1",
                name="Org",
                slug="worker-org",
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
        session.add(
            Event(
                id="evt-worker",
                organization_id="org-1",
                user_id="alice",
                title="Worker event",
            )
        )
        run = Run(
            organization_id="org-1",
            user_id="alice",
            event_id="evt-worker",
            agent_id="agent-1",
            session_id="session-1",
            task="execute",
            status="running",
            attempt_count=1,
            lease_owner="worker-a",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        session.add(run)
        await session.commit()
        run_id = run.id

    await execute_claimed_run(run_id, "worker-a")

    async with db_session_maker() as session:
        completed = await session.get(Run, run_id)
        assert completed.status == "completed"
        assert completed.lease_owner is None
        assert completed.prompt_fingerprint == "prompt-fp"
        assert completed.selected_skills == ["teach@1"]
        assert completed.artifacts == [
            {
                "kind": "file",
                "tool_name": "write_file",
                "path": "lessons/division.html",
            }
        ]
        assert completed.plan == [
            {
                "id": "1",
                "content": "answer the user",
                "status": "completed",
            }
        ]
        traces = (await session.execute(select(TraceRecord))).scalars().all()
        assert len(traces) == 1
        assert traces[0].organization_id == "org-1"


async def test_worker_rejects_run_after_membership_revocation(
    db_session_maker, monkeypatch
):
    monkeypatch.setattr("aloy_backend.background.async_session", db_session_maker)
    invoked = False

    def should_not_build(**kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("model path must not run")

    monkeypatch.setattr("aloy_backend.background.build_orchestrator", should_not_build)
    async with db_session_maker() as session:
        session.add(
            Organization(
                id="org-2",
                name="Org 2",
                slug="revoked-org",
                created_by="alice",
                policy={},
            )
        )
        session.add(
            OrganizationMembership(
                organization_id="org-2",
                user_id="alice",
                role="member",
                status="suspended",
            )
        )
        run = Run(
            organization_id="org-2",
            user_id="alice",
            event_id="evt-worker-org-2",
            agent_id="agent-1",
            session_id="session-1",
            task="blocked",
            status="running",
            attempt_count=1,
            max_attempts=1,
            lease_owner="worker-a",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        session.add(run)
        await session.commit()
        run_id = run.id

    await execute_claimed_run(run_id, "worker-a")

    async with db_session_maker() as session:
        failed = await session.get(Run, run_id)
        assert failed.status == "failed"
        assert failed.lease_owner is None
    assert invoked is False


async def test_worker_stages_external_action_without_executing(
    db_session_maker, monkeypatch
):
    monkeypatch.setattr("aloy_backend.background.async_session", db_session_maker)
    tool_calls = []
    staged = {}

    class SendParams(BaseModel):
        to: str

    registry = ToolRegistry()
    registry.register_tool(
        "gmail_send",
        SendParams,
        lambda params, context: tool_calls.append(params),
        "send",
    )

    class FakeOrchestrator:
        tools_registry = registry

        async def execute_task(self, **kwargs):
            response = await kwargs["hitl_handler"].request_approval(
                ApprovalRequest(
                    action_requests=[
                        ActionRequest(
                            name="gmail_send",
                            arguments={"to": "a@b.com"},
                            description="Send the update",
                        )
                    ],
                    review_configs=[
                        ReviewConfig(
                            action_name="gmail_send",
                            allowed_decisions=["approve", "reject"],
                        )
                    ],
                    task_id="task-1",
                    step_number=1,
                )
            )
            staged.update(response.decisions[0].result or {})
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
    async with db_session_maker() as session:
        session.add(
            Organization(
                id="org-stage-worker",
                name="Stage worker",
                slug="stage-worker",
                created_by="alice",
                policy={},
            )
        )
        session.add(
            OrganizationMembership(
                organization_id="org-stage-worker",
                user_id="alice",
                role="member",
            )
        )
        session.add(
            Event(
                id="evt-stage-worker",
                organization_id="org-stage-worker",
                user_id="alice",
                title="Stage worker",
            )
        )
        run = Run(
            organization_id="org-stage-worker",
            user_id="alice",
            event_id="evt-stage-worker",
            agent_id="agent-1",
            session_id="session-1",
            task="send update",
            status="running",
            attempt_count=1,
            lease_owner="worker-a",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        session.add(run)
        await session.commit()
        run_id = run.id

    await execute_claimed_run(run_id, "worker-a")

    assert staged["status"] == "staged"
    assert tool_calls == []
    async with db_session_maker() as session:
        completed = await session.get(Run, run_id)
        proposal = await session.get(ActionProposal, staged["proposal_id"])
        assert completed.status == "completed"
        assert proposal.status == "pending"
        assert proposal.origin_run_id == run_id
