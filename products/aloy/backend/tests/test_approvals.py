"""HITL approval bridge — the run pauses on a consequential tool, emits an
approval_request frame, and blocks until the user decides. Mirrors the clarify
bridge's pause/resume + ownership-scoped resolve."""

import asyncio

import pytest

from aloy_backend.approvals import (
    APPROVAL_BRIDGES,
    ApprovalBridge,
    build_write_hitl_config,
    resolve_approval,
)
from pori import ActionRequest, ApprovalRequest, ReviewConfig

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
