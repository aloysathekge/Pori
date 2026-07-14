"""Interactive HITL approval — the Proposal-commit rail.

When the agent is about to run a consequential tool (send an email, etc.), the
kernel's HITL gate calls ``request_approval``; this bridge emits an
``approval_request`` frame to the client and BLOCKS the run until the user
approves or rejects via the approve endpoint. It is the twin of the clarify
bridge (streaming.py): same emit-frame + await-future + ownership-scoped resolve
plumbing, so a decision only ever resolves the caller's own paused run.

The run executes on a worker thread (asyncio.run inside run_in_executor), so
``request_approval`` awaits a future created on that worker loop; the endpoint
resolves it from the serving loop via ``call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable, Dict, Iterable, List

from pori import (
    ApprovalRequest,
    ApprovalResponse,
    Decision,
    EditedAction,
    HITLConfig,
    HITLHandler,
    InterruptConfig,
)

# Active approval bridges, keyed to the (organization_id, user_id) that owns the
# run — mirrors streaming.CLARIFY_BRIDGES. A decision for an id owned by another
# user resolves to False (404), never touching their run.
APPROVAL_BRIDGES: Dict["ApprovalBridge", tuple] = {}


def build_write_hitl_config(write_tools: Iterable[str]) -> HITLConfig:
    """Gate the given consequential write tools behind approve/reject. External
    actions (sending email, etc.) always Ask — the vision's fixed-policy floor.
    Each action is decided on its own (no duplicate auto-approve): a second send
    is a second consequence, not a repeat of the first."""
    tools = tuple(write_tools)
    return HITLConfig(
        enabled=bool(tools),
        interrupt_on={
            name: InterruptConfig(allowed_decisions=["approve", "reject"])
            for name in tools
        },
        auto_approve_duplicates=False,
    )


def _to_event(approval_id: str, request: ApprovalRequest) -> Dict[str, Any]:
    """The on-the-wire shape the client renders as an approval card. V1 reviews
    one action at a time (the loop pauses per gated call)."""
    action = request.action_requests[0]
    review = request.review_configs[0]
    return {
        "type": "approval_request",
        "id": approval_id,
        "tool": action.name,
        "arguments": action.arguments,
        "description": action.description,
        "allowed_decisions": list(review.allowed_decisions),
    }


def _to_decision(payload: Dict[str, Any]) -> Decision:
    decision_type = payload.get("type", "reject")
    edited = None
    if decision_type == "edit" and isinstance(payload.get("edited_action"), dict):
        ea = payload["edited_action"]
        edited = EditedAction(name=ea.get("name", ""), args=ea.get("args", {}) or {})
    return Decision(
        type=decision_type, edited_action=edited, message=payload.get("message")
    )


class ApprovalBridge(HITLHandler):
    def __init__(self, emit: Callable[[Dict[str, Any]], Any]):
        self._emit = emit
        self._pending: Dict[str, "asyncio.Future[list]"] = {}

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        approval_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[list]" = loop.create_future()
        self._pending[approval_id] = future
        self._emit(_to_event(approval_id, request))
        try:
            raw = await future  # list of decision dicts delivered by the endpoint
        finally:
            self._pending.pop(approval_id, None)
        decisions = [_to_decision(d) for d in raw] or [Decision(type="reject")]
        return ApprovalResponse(decisions=decisions)

    def submit_decisions(self, approval_id: str, decisions: List[dict]) -> bool:
        """Deliver the user's decision(s). Thread-safe; False for an unknown or
        already-resolved id."""
        future = self._pending.get(approval_id)
        if future is not None and not future.done():
            future.get_loop().call_soon_threadsafe(future.set_result, decisions)
            return True
        return False

    def cancel_pending(self) -> None:
        """On stop / client disconnect, resolve any outstanding approval as
        REJECT — the safe default: never deliver a consequential action without
        an explicit yes."""
        for approval_id in list(self._pending):
            self.submit_decisions(
                approval_id,
                [{"type": "reject", "message": "Run ended before approval"}],
            )

    def pending_ids(self) -> List[str]:
        return list(self._pending)


def resolve_approval(
    approval_id: str,
    decisions: List[dict],
    *,
    organization_id: str,
    user_id: str,
) -> bool:
    """Deliver a decision to the CALLER'S awaiting run only. An approval id
    owned by another user resolves to False (404 at the endpoint)."""
    owner = (organization_id, user_id)
    for bridge, bridge_owner in list(APPROVAL_BRIDGES.items()):
        if bridge_owner != owner:
            continue
        if bridge.submit_decisions(approval_id, decisions):
            return True
    return False
