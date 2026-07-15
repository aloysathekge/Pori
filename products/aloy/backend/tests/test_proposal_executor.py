from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from sqlmodel import select

import aloy_backend.proposal_executor as executor_module
import aloy_backend.routes.events as events_route
import aloy_backend.worker as worker_module
from aloy_backend.models import (
    ActionProposal,
    Event,
    EventTrailEntry,
    Organization,
    OrganizationMembership,
)
from aloy_backend.proposal_executor import (
    ProposalDecisionError,
    decide_proposal,
    execute_proposal,
    expire_due_proposals,
    reconcile_stale_executions,
)
from pori import stable_fingerprint
from pori.tools.registry import ToolRegistry

pytestmark = pytest.mark.asyncio


class SendParams(BaseModel):
    to: str
    subject: str = "Status"


def _registry(calls: list[dict], *, delay: float = 0) -> ToolRegistry:
    def send(params: SendParams, context: dict) -> dict:
        if delay:
            time.sleep(delay)
        calls.append(
            {
                "to": params.to,
                "attempt": context.get("execution_attempt_id"),
            }
        )
        return {"sent": True, "id": "provider-message-1", "to": params.to}

    registry = ToolRegistry()
    registry.register_tool("gmail_send", SendParams, send, "send")
    return registry


async def _seed(
    db_session_maker,
    registry: ToolRegistry,
    *,
    proposal_id: str,
    status: str = "pending",
    membership_status: str = "active",
    fingerprint: str | None = None,
    expires_at: datetime | None = None,
) -> ActionProposal:
    tool = registry.get_tool("gmail_send")
    proposal = ActionProposal(
        id=proposal_id,
        organization_id="org-proposal",
        user_id="alice",
        event_id="evt-proposal",
        origin_session_id="session-1",
        origin_run_id="run-1",
        tool="gmail_send",
        args={"to": "team@example.com", "subject": "Status"},
        tool_schema_fingerprint=fingerprint
        or stable_fingerprint(tool.param_model.model_json_schema()),
        reason="Send the update",
        impact="Sends an email",
        risk="high",
        routing="ask",
        status=status,
        expires_at=expires_at or datetime.now(timezone.utc) + timedelta(days=1),
        safe_default={"decision": "reject"},
    )
    async with db_session_maker() as session:
        session.add_all(
            [
                Organization(
                    id="org-proposal",
                    name="Proposal Org",
                    slug="proposal-org",
                    created_by="alice",
                    policy={},
                ),
                OrganizationMembership(
                    organization_id="org-proposal",
                    user_id="alice",
                    role="owner",
                    status=membership_status,
                ),
                Event(
                    id="evt-proposal",
                    organization_id="org-proposal",
                    user_id="alice",
                    title="Proposal Event",
                ),
                proposal,
            ]
        )
        await session.commit()
    return proposal


@pytest.fixture
def available_surface(monkeypatch):
    async def resolve(*args, **kwargs):
        return SimpleNamespace(
            denied_tools=(),
            tool_context_extra={"connections": {"google": {"access_token": "test"}}},
        )

    monkeypatch.setattr(executor_module, "resolve_run_surface", resolve)


async def test_approve_executes_once_and_commits_receipt_and_trail(
    db_session_maker, available_surface
):
    calls: list[dict] = []
    registry = _registry(calls)
    proposal = await _seed(db_session_maker, registry, proposal_id="prop-commit")
    async with db_session_maker() as session:
        decided = await decide_proposal(
            session,
            event_id=proposal.event_id,
            proposal_id=proposal.id,
            organization_id=proposal.organization_id,
            user_id=proposal.user_id,
            actor_id="alice",
            decision="approve",
            registry=registry,
        )
        assert decided.status == "approved"

    result = await execute_proposal(
        proposal.id,
        session_factory=db_session_maker,
        registry=registry,
    )

    assert result.status == "committed" and result.claimed is True
    assert len(calls) == 1 and calls[0]["attempt"].startswith("attempt_")
    async with db_session_maker() as session:
        stored = await session.get(ActionProposal, proposal.id)
        trails = (
            (
                await session.execute(
                    select(EventTrailEntry)
                    .where(EventTrailEntry.proposal_id == proposal.id)
                    .order_by(EventTrailEntry.created_at)
                )
            )
            .scalars()
            .all()
        )
        assert stored.status == "committed"
        assert stored.provider_operation_id == "provider-message-1"
        assert stored.receipt["status"] == "succeeded"
        assert [entry.kind for entry in trails] == [
            "proposal_decided",
            "proposal_committed",
        ]
        assert trails[-1].evidence_refs[0]["receipt_id"].startswith("rcpt_")


async def test_concurrent_executor_calls_have_one_claim_and_one_provider_call(
    db_session_maker, available_surface
):
    calls: list[dict] = []
    registry = _registry(calls, delay=0.05)
    proposal = await _seed(
        db_session_maker,
        registry,
        proposal_id="prop-single-claim",
        status="approved",
    )

    results = await asyncio.gather(
        execute_proposal(
            proposal.id, session_factory=db_session_maker, registry=registry
        ),
        execute_proposal(
            proposal.id, session_factory=db_session_maker, registry=registry
        ),
    )

    assert len(calls) == 1
    assert sum(result.claimed for result in results) == 1
    async with db_session_maker() as session:
        assert (await session.get(ActionProposal, proposal.id)).status == "committed"


async def test_concurrent_decisions_have_one_winner(db_session_maker):
    registry = _registry([])
    proposal = await _seed(
        db_session_maker, registry, proposal_id="prop-single-decision"
    )

    async def decide(decision):
        async with db_session_maker() as session:
            try:
                result = await decide_proposal(
                    session,
                    event_id=proposal.event_id,
                    proposal_id=proposal.id,
                    organization_id=proposal.organization_id,
                    user_id=proposal.user_id,
                    actor_id=f"alice:{decision}",
                    decision=decision,
                    registry=registry,
                )
                return result.status
            except ProposalDecisionError:
                return "conflict"

    outcomes = await asyncio.gather(decide("approve"), decide("reject"))

    assert outcomes.count("conflict") == 1
    async with db_session_maker() as session:
        trails = (
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.proposal_id == proposal.id
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(trails) == 1


async def test_schema_drift_fails_before_provider_call(
    db_session_maker, available_surface
):
    calls: list[dict] = []
    registry = _registry(calls)
    proposal = await _seed(
        db_session_maker,
        registry,
        proposal_id="prop-schema-drift",
        status="approved",
        fingerprint="stale-schema",
    )

    result = await execute_proposal(
        proposal.id, session_factory=db_session_maker, registry=registry
    )

    assert result.status == "failed"
    assert calls == []
    async with db_session_maker() as session:
        stored = await session.get(ActionProposal, proposal.id)
        assert stored.status == "failed"
        assert "schema changed" in stored.error.lower()


async def test_execution_time_membership_revocation_fails_safely(
    db_session_maker, available_surface
):
    calls: list[dict] = []
    registry = _registry(calls)
    proposal = await _seed(
        db_session_maker,
        registry,
        proposal_id="prop-revoked",
        status="approved",
        membership_status="revoked",
    )

    result = await execute_proposal(
        proposal.id, session_factory=db_session_maker, registry=registry
    )

    assert result.status == "failed"
    assert calls == []


async def test_execution_time_policy_revocation_fails_safely(
    db_session_maker, available_surface
):
    calls: list[dict] = []
    registry = _registry(calls)
    proposal = await _seed(
        db_session_maker,
        registry,
        proposal_id="prop-policy-revoked",
        status="approved",
    )
    async with db_session_maker() as session:
        organization = await session.get(Organization, proposal.organization_id)
        organization.policy = {"denied_tools": ["gmail_send"]}
        session.add(organization)
        await session.commit()

    result = await execute_proposal(
        proposal.id, session_factory=db_session_maker, registry=registry
    )

    assert result.status == "failed"
    assert calls == []


async def test_execution_time_credential_removal_fails_safely(
    db_session_maker, monkeypatch
):
    calls: list[dict] = []
    registry = _registry(calls)
    proposal = await _seed(
        db_session_maker,
        registry,
        proposal_id="prop-credentials-revoked",
        status="approved",
    )

    async def unavailable(*args, **kwargs):
        return SimpleNamespace(
            denied_tools=("gmail_send",),
            tool_context_extra={"connections": {}},
        )

    monkeypatch.setattr(executor_module, "resolve_run_surface", unavailable)
    result = await execute_proposal(
        proposal.id, session_factory=db_session_maker, registry=registry
    )

    assert result.status == "failed"
    assert calls == []


async def test_expired_pending_proposal_applies_reject_safe_default(
    db_session_maker,
):
    calls: list[dict] = []
    registry = _registry(calls)
    now = datetime.now(timezone.utc)
    proposal = await _seed(
        db_session_maker,
        registry,
        proposal_id="prop-expired",
        expires_at=now - timedelta(seconds=1),
    )

    assert await expire_due_proposals(session_factory=db_session_maker, now=now) == 1
    async with db_session_maker() as session:
        stored = await session.get(ActionProposal, proposal.id)
        assert stored.status == "withdrawn"
        assert stored.decided_by == "system:expiry"
        trail = (
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.proposal_id == proposal.id
                    )
                )
            )
            .scalars()
            .one()
        )
        assert trail.payload["reason"] == "expired"
    assert calls == []


async def test_receipt_persistence_crash_becomes_indeterminate_without_retry(
    db_session_maker, available_surface, monkeypatch
):
    calls: list[dict] = []
    registry = _registry(calls)
    proposal = await _seed(
        db_session_maker,
        registry,
        proposal_id="prop-crash-window",
        status="approved",
    )

    async def fail_finalize(*args, **kwargs):
        raise RuntimeError("simulated commit loss")

    monkeypatch.setattr(executor_module, "_finalize_execution", fail_finalize)
    first = await execute_proposal(
        proposal.id, session_factory=db_session_maker, registry=registry
    )
    second = await execute_proposal(
        proposal.id, session_factory=db_session_maker, registry=registry
    )

    assert first.status == "indeterminate"
    assert second.claimed is False and second.status == "indeterminate"
    assert len(calls) == 1
    async with db_session_maker() as session:
        assert (await session.get(ActionProposal, proposal.id)).status == (
            "indeterminate"
        )


async def test_stale_execution_claim_reconciles_to_indeterminate(db_session_maker):
    registry = _registry([])
    proposal = await _seed(
        db_session_maker,
        registry,
        proposal_id="prop-stale-claim",
        status="executing",
    )
    now = datetime.now(timezone.utc)
    async with db_session_maker() as session:
        stored = await session.get(ActionProposal, proposal.id)
        stored.execution_attempt_id = "attempt-abandoned"
        stored.updated_at = now - timedelta(minutes=10)
        session.add(stored)
        await session.commit()

    assert (
        await reconcile_stale_executions(
            session_factory=db_session_maker,
            now=now,
        )
        == 1
    )
    async with db_session_maker() as session:
        stored = await session.get(ActionProposal, proposal.id)
        assert stored.status == "indeterminate"
        assert "requires review" in stored.error


async def test_edit_revalidates_same_tool_and_requires_separate_approval(
    db_session_maker,
):
    calls: list[dict] = []
    registry = _registry(calls)
    proposal = await _seed(db_session_maker, registry, proposal_id="prop-edit")

    async with db_session_maker() as session:
        with pytest.raises(ProposalDecisionError, match="cannot change the tool"):
            await decide_proposal(
                session,
                event_id=proposal.event_id,
                proposal_id=proposal.id,
                organization_id=proposal.organization_id,
                user_id=proposal.user_id,
                actor_id="alice",
                decision="edit",
                edited_action={"name": "calendar_create_event", "args": {}},
                registry=registry,
            )
        await session.rollback()

    async with db_session_maker() as session:
        edited = await decide_proposal(
            session,
            event_id=proposal.event_id,
            proposal_id=proposal.id,
            organization_id=proposal.organization_id,
            user_id=proposal.user_id,
            actor_id="alice",
            decision="edit",
            edited_action={
                "name": "gmail_send",
                "args": {"to": "new@example.com"},
            },
            registry=registry,
        )
        assert edited.status == "pending"
        assert edited.args == {"to": "new@example.com", "subject": "Status"}


async def test_event_decision_endpoint_rejects_tenant_scoped_proposal(
    client, db_session_maker
):
    conversation = await client.post("/v1/conversations", json={"title": "Decision"})
    assert conversation.status_code == 201
    event_id = conversation.json()["event_id"]
    registry = _registry([])
    tool = registry.get_tool("gmail_send")
    proposal = ActionProposal(
        id="prop-api-reject",
        organization_id="user:test-user",
        user_id="test-user",
        event_id=event_id,
        tool="gmail_send",
        args={"to": "team@example.com", "subject": "Status"},
        tool_schema_fingerprint=stable_fingerprint(
            tool.param_model.model_json_schema()
        ),
        reason="Send",
        impact="Email",
        risk="high",
        routing="ask",
        status="pending",
    )
    async with db_session_maker() as session:
        session.add(proposal)
        await session.commit()

    denied = await client.post(
        f"/v1/events/{event_id}/proposals/{proposal.id}/decision",
        json={"decision": "reject"},
        headers={"X-Test-User": "other-user"},
    )
    assert denied.status_code == 404
    response = await client.post(
        f"/v1/events/{event_id}/proposals/{proposal.id}/decision",
        json={"decision": "reject"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "withdrawn"
    async with db_session_maker() as session:
        assert (await session.get(ActionProposal, proposal.id)).status == "withdrawn"


async def test_event_approve_endpoint_enqueues_the_executor(
    client, db_session_maker, monkeypatch
):
    conversation = await client.post("/v1/conversations", json={"title": "Approve"})
    event_id = conversation.json()["event_id"]
    registry = _registry([])
    proposal = ActionProposal(
        id="prop-api-approve",
        organization_id="user:test-user",
        user_id="test-user",
        event_id=event_id,
        tool="gmail_send",
        args={"to": "team@example.com", "subject": "Status"},
        tool_schema_fingerprint=stable_fingerprint(
            registry.get_tool("gmail_send").param_model.model_json_schema()
        ),
        reason="Send",
        impact="Email",
        risk="high",
        routing="ask",
        status="pending",
    )
    async with db_session_maker() as session:
        session.add(proposal)
        await session.commit()
    enqueued: list[str] = []

    async def fake_execute(proposal_id, **kwargs):
        enqueued.append(proposal_id)

    monkeypatch.setattr(events_route, "execute_proposal", fake_execute)
    response = await client.post(
        f"/v1/events/{event_id}/proposals/{proposal.id}/decision",
        json={"decision": "approve"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    assert enqueued == [proposal.id]


async def test_conversation_approval_alias_resolves_durable_proposal(
    client, db_session_maker
):
    conversation = await client.post("/v1/conversations", json={"title": "Alias"})
    event_id = conversation.json()["event_id"]
    registry = _registry([])
    proposal = ActionProposal(
        id="prop-alias-reject",
        organization_id="user:test-user",
        user_id="test-user",
        event_id=event_id,
        tool="gmail_send",
        args={"to": "team@example.com", "subject": "Status"},
        tool_schema_fingerprint=stable_fingerprint(
            registry.get_tool("gmail_send").param_model.model_json_schema()
        ),
        reason="Send",
        impact="Email",
        risk="high",
        routing="ask",
        status="pending",
    )
    async with db_session_maker() as session:
        session.add(proposal)
        await session.commit()

    response = await client.post(
        f"/v1/conversations/approve/{proposal.id}",
        json={"decisions": [{"type": "reject"}]},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "withdrawn"


async def test_worker_tick_processes_approved_proposals_before_runs(monkeypatch):
    calls: list[str] = []

    async def expire():
        calls.append("expire")

    async def reconcile():
        calls.append("reconcile")

    async def execute_next():
        calls.append("proposal")
        return SimpleNamespace(status="committed")

    async def claim_run(worker_id):
        calls.append("run")
        return None

    monkeypatch.setattr(worker_module, "expire_due_proposals", expire)
    monkeypatch.setattr(worker_module, "reconcile_stale_executions", reconcile)
    monkeypatch.setattr(worker_module, "execute_next_approved_proposal", execute_next)
    monkeypatch.setattr(worker_module, "claim_next_run", claim_run)

    assert await worker_module.run_once("worker-1") is True
    assert calls == ["expire", "reconcile", "proposal"]
