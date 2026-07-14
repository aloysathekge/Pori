"""HITL approval bridge — the run pauses on a consequential tool, emits an
approval_request frame, and blocks until the user decides. Mirrors the clarify
bridge's pause/resume + ownership-scoped resolve."""

import asyncio

import pytest
from pydantic import BaseModel, Field
from sqlmodel import select

from aloy_backend.approvals import (
    APPROVAL_BRIDGES,
    ApprovalBridge,
    NonInteractiveDenyHandler,
    ProposalStagingHandler,
    build_write_hitl_config,
    non_interactive_write_gate,
    proposal_write_gate,
    resolve_approval,
)
from aloy_backend.models import ActionProposal, Event, EventTrailEntry
from pori import ActionRequest, ApprovalRequest, ReviewConfig, RunContext
from pori.tools.registry import ToolRegistry

pytestmark = pytest.mark.asyncio


def _request(tool: str = "gmail_send") -> ApprovalRequest:
    return ApprovalRequest(
        action_requests=[
            ActionRequest(name=tool, arguments={"to": "a@b.com"}, description="Send")
        ],
        review_configs=[
            ReviewConfig(action_name=tool, allowed_decisions=["approve", "reject"])
        ],
        task_id="t1",
        step_number=1,
    )


class TestBuildConfig:
    def test_gates_the_write_tools(self):
        cfg = build_write_hitl_config(["gmail_send", "gmail_send_draft"])
        assert cfg.enabled is True
        assert set(cfg.interrupt_on) == {"gmail_send", "gmail_send_draft"}
        assert cfg.auto_approve_duplicates is False

    def test_empty_disables(self):
        assert build_write_hitl_config([]).enabled is False


class TestApprovalBridge:
    async def test_emits_frame_then_resolves_on_approve(self):
        frames = []
        bridge = ApprovalBridge(emit=frames.append)

        task = asyncio.create_task(bridge.request_approval(_request()))
        await asyncio.sleep(0)  # let request_approval emit + start awaiting

        assert len(frames) == 1
        event = frames[0]
        assert event["type"] == "approval_request"
        assert event["tool"] == "gmail_send"
        assert event["allowed_decisions"] == ["approve", "reject"]
        approval_id = event["id"]

        assert bridge.submit_decisions(approval_id, [{"type": "approve"}]) is True
        response = await task
        assert response.decisions[0].type == "approve"
        assert bridge.pending_ids() == []

    async def test_reject_carries_message(self):
        frames = []
        bridge = ApprovalBridge(emit=frames.append)
        task = asyncio.create_task(bridge.request_approval(_request()))
        await asyncio.sleep(0)
        bridge.submit_decisions(
            frames[0]["id"], [{"type": "reject", "message": "not now"}]
        )
        response = await task
        assert response.decisions[0].type == "reject"
        assert response.decisions[0].message == "not now"

    async def test_cancel_pending_rejects_by_default(self):
        """A stopped/disconnected run must never send — outstanding approvals
        resolve to reject."""
        frames = []
        bridge = ApprovalBridge(emit=frames.append)
        task = asyncio.create_task(bridge.request_approval(_request()))
        await asyncio.sleep(0)
        bridge.cancel_pending()
        response = await task
        assert response.decisions[0].type == "reject"

    async def test_unknown_id_returns_false(self):
        bridge = ApprovalBridge(emit=lambda e: None)
        assert bridge.submit_decisions("nope", [{"type": "approve"}]) is False

    async def test_enrich_adds_display_detail(self):
        """A send_draft gate carries only a draft id; the enricher fills the
        real email so the card shows it."""
        frames = []

        def enrich(tool, args):
            assert tool == "gmail_send_draft" and args["draft_id"] == "d1"
            return {"to": "a@b.com", "subject": "Hi", "body": "hello"}

        bridge = ApprovalBridge(emit=frames.append, enrich=enrich)
        req = ApprovalRequest(
            action_requests=[
                ActionRequest(
                    name="gmail_send_draft",
                    arguments={"draft_id": "d1"},
                    description="",
                )
            ],
            review_configs=[
                ReviewConfig(
                    action_name="gmail_send_draft", allowed_decisions=["approve"]
                )
            ],
            task_id="t1",
            step_number=1,
        )
        task = asyncio.create_task(bridge.request_approval(req))
        await asyncio.sleep(0)
        args = frames[0]["arguments"]
        assert args["to"] == "a@b.com" and args["subject"] == "Hi"
        assert args["body"] == "hello" and args["draft_id"] == "d1"
        bridge.submit_decisions(frames[0]["id"], [{"type": "approve"}])
        await task


class TestProposalStaging:
    async def test_valid_action_stages_proposal_and_trail_without_execution(
        self, db_session_maker
    ):
        calls = []

        class SendParams(BaseModel):
            to: str
            retries: int = Field(default=1, ge=1)

        def send_tool(params, context):
            calls.append(params)
            return {"sent": True}

        registry = ToolRegistry()
        registry.register_tool("gmail_send", SendParams, send_tool, "send")
        async with db_session_maker() as session:
            session.add(
                Event(
                    id="evt-stage",
                    organization_id="org-1",
                    user_id="alice",
                    title="Stage",
                )
            )
            await session.commit()

        frames = []
        handler = ProposalStagingHandler(
            run_context=RunContext(
                organization_id="org-1",
                user_id="alice",
                agent_id="agent-1",
                session_id="session-1",
                run_id="run-1",
                event_id="evt-stage",
            ),
            tools_registry=registry,
            emit=frames.append,
            session_factory=db_session_maker,
        )

        response = await handler.request_approval(_request())

        decision = response.decisions[0]
        assert decision.type == "defer"
        assert decision.result["status"] == "staged"
        assert calls == []
        assert frames[0]["proposal_id"] == decision.result["proposal_id"]
        async with db_session_maker() as session:
            proposal = await session.get(ActionProposal, decision.result["proposal_id"])
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
            assert proposal.status == "pending"
            assert proposal.args == {"to": "a@b.com", "retries": 1}
            assert proposal.routing == "ask"
            assert proposal.safe_default == {"decision": "reject"}
            assert trail.kind == "proposal_staged"
            assert trail.run_id == "run-1"

    async def test_invalid_payload_creates_no_proposal(self, db_session_maker):
        class SendParams(BaseModel):
            to: str
            retries: int = Field(ge=1)

        registry = ToolRegistry()
        registry.register_tool("gmail_send", SendParams, lambda p, c: None, "send")
        async with db_session_maker() as session:
            session.add(
                Event(
                    id="evt-invalid",
                    organization_id="org-1",
                    user_id="alice",
                    title="Invalid",
                )
            )
            await session.commit()

        handler = ProposalStagingHandler(
            run_context=RunContext(
                organization_id="org-1",
                user_id="alice",
                agent_id="agent-1",
                session_id="session-1",
                run_id="run-1",
                event_id="evt-invalid",
            ),
            tools_registry=registry,
            session_factory=db_session_maker,
        )
        invalid = _request()
        invalid.action_requests[0].arguments = {"to": "a@b.com", "retries": 0}

        response = await handler.request_approval(invalid)

        assert response.decisions[0].type == "reject"
        async with db_session_maker() as session:
            proposals = (await session.execute(select(ActionProposal))).scalars().all()
            trails = (await session.execute(select(EventTrailEntry))).scalars().all()
            assert proposals == []
            assert trails == []

    async def test_worker_thread_marshals_persistence_to_serving_loop(
        self, db_session_maker
    ):
        class SendParams(BaseModel):
            to: str

        registry = ToolRegistry()
        registry.register_tool("gmail_send", SendParams, lambda p, c: None, "send")
        async with db_session_maker() as session:
            session.add(
                Event(
                    id="evt-threaded",
                    organization_id="org-1",
                    user_id="alice",
                    title="Threaded",
                )
            )
            await session.commit()

        serving_loop = asyncio.get_running_loop()
        handler = ProposalStagingHandler(
            run_context=RunContext(
                organization_id="org-1",
                user_id="alice",
                agent_id="agent-1",
                session_id="session-1",
                run_id="run-threaded",
                event_id="evt-threaded",
            ),
            tools_registry=registry,
            session_factory=db_session_maker,
            owner_loop=serving_loop,
        )

        response = await serving_loop.run_in_executor(
            None, lambda: asyncio.run(handler.request_approval(_request()))
        )

        assert response.decisions[0].type == "defer"
        async with db_session_maker() as session:
            proposal = await session.get(
                ActionProposal, response.decisions[0].result["proposal_id"]
            )
            assert proposal.origin_run_id == "run-threaded"

    async def test_enrich_failure_is_swallowed(self):
        """Enrichment is best-effort — a raising enricher must not block the gate."""
        frames = []
        bridge = ApprovalBridge(
            emit=frames.append,
            enrich=lambda t, a: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        task = asyncio.create_task(bridge.request_approval(_request("gmail_send")))
        await asyncio.sleep(0)
        assert frames[0]["arguments"] == {"to": "a@b.com"}  # original args intact
        bridge.submit_decisions(frames[0]["id"], [{"type": "approve"}])
        await task


class TestNonInteractiveGuardrail:
    async def test_denies_every_gated_action(self):
        """A run with no user attached must REJECT a gated tool, not run it."""
        handler = NonInteractiveDenyHandler()
        response = await handler.request_approval(_request("gmail_send"))
        assert response.decisions[0].type == "reject"
        assert "approval" in (response.decisions[0].message or "").lower()

    def test_gate_helper_denies_the_write_tools(self):
        handler, config = non_interactive_write_gate()
        assert isinstance(handler, NonInteractiveDenyHandler)
        assert config.enabled is True
        # The consequential Gmail writes are gated (so they hit the deny).
        assert "gmail_send" in config.interrupt_on
        assert "gmail_send_draft" in config.interrupt_on

    def test_proposal_gate_replaces_deny_for_event_runs(self):
        registry = ToolRegistry()
        handler, config = proposal_write_gate(
            run_context=RunContext(
                organization_id="org-1",
                user_id="alice",
                agent_id="agent-1",
                session_id="session-1",
                run_id="run-1",
                event_id="evt-1",
            ),
            tools_registry=registry,
        )
        assert isinstance(handler, ProposalStagingHandler)
        assert "gmail_send" in config.interrupt_on


class TestResolveApprovalOwnership:
    async def test_only_owner_can_resolve(self):
        frames = []
        bridge = ApprovalBridge(emit=frames.append)
        APPROVAL_BRIDGES[bridge] = ("org-1", "alice")
        try:
            task = asyncio.create_task(bridge.request_approval(_request()))
            await asyncio.sleep(0)
            approval_id = frames[0]["id"]

            # Wrong owner → not resolved (would 404 at the endpoint).
            assert (
                resolve_approval(
                    approval_id,
                    [{"type": "approve"}],
                    organization_id="org-2",
                    user_id="bob",
                )
                is False
            )
            # Correct owner → resolved.
            assert (
                resolve_approval(
                    approval_id,
                    [{"type": "approve"}],
                    organization_id="org-1",
                    user_id="alice",
                )
                is True
            )
            response = await task
            assert response.decisions[0].type == "approve"
        finally:
            APPROVAL_BRIDGES.pop(bridge, None)
