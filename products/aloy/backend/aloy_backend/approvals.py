"""Proposal staging for consequential tools, plus the legacy approval bridge.

Event-aware Aloy runs use :class:`ProposalStagingHandler`: validate the tool
payload, atomically persist Proposal + Trail, emit the durable proposal id, and
return a kernel ``defer`` decision. The originating run never executes or waits
for the external consequence. ``ApprovalBridge`` remains temporarily for
kernel-only callers that have no Event aggregate.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from pydantic import ValidationError

from pori import (
    ApprovalRequest,
    ApprovalResponse,
    Decision,
    EditedAction,
    HITLConfig,
    HITLHandler,
    InterruptConfig,
    RunContext,
    stable_fingerprint,
)

from .database import async_session
from .models import ActionProposal, Event, EventTrailEntry

logger = logging.getLogger("aloy_backend")

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


def _to_event(
    approval_id: str,
    request: ApprovalRequest,
    enrich: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """The on-the-wire shape the client renders as an approval card. V1 reviews
    one action at a time (the loop pauses per gated call).

    ``enrich`` adds display detail the tool call itself lacks — e.g. a
    ``gmail_send_draft`` carries only a draft id, so the enricher fetches the
    draft's to/subject/body so the user reviews the real email, not an id."""
    action = request.action_requests[0]
    review = request.review_configs[0]
    arguments: Dict[str, Any] = dict(action.arguments)
    if enrich is not None:
        try:
            arguments.update(enrich(action.name, arguments) or {})
        except Exception:  # enrichment is best-effort — never block the gate
            pass
    return {
        "type": "approval_request",
        "id": approval_id,
        "tool": action.name,
        "arguments": arguments,
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
    def __init__(
        self,
        emit: Callable[[Dict[str, Any]], Any],
        enrich: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        self._emit = emit
        self._enrich = enrich
        self._pending: Dict[str, "asyncio.Future[list]"] = {}

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        approval_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[list]" = loop.create_future()
        self._pending[approval_id] = future
        self._emit(_to_event(approval_id, request, self._enrich))
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


def _proposal_impact(tool_name: str) -> str:
    if tool_name.startswith("gmail_send"):
        return "Will send an external email if later approved and committed."
    if tool_name == "calendar_create_event":
        return "Will create an external calendar event if later approved and committed."
    return "Will perform an external action if later approved and committed."


class ProposalStagingHandler(HITLHandler):
    """Validate and durably stage gated calls without executing them."""

    def __init__(
        self,
        *,
        run_context: RunContext,
        tools_registry: Any,
        emit: Optional[Callable[[Dict[str, Any]], Any]] = None,
        enrich: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
        session_factory: Any = async_session,
        owner_loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self._run_context = run_context
        self._tools_registry = tools_registry
        self._emit = emit
        self._enrich = enrich
        self._session_factory = session_factory
        self._owner_loop = owner_loop

    async def _on_owner_loop(self, coroutine):
        current = asyncio.get_running_loop()
        if self._owner_loop is None or self._owner_loop is current:
            return await coroutine
        future = asyncio.run_coroutine_threadsafe(coroutine, self._owner_loop)
        return await asyncio.wrap_future(future)

    async def _stage(self, request: ApprovalRequest, index: int) -> Decision:
        action = request.action_requests[index]
        try:
            tool = self._tools_registry.get_tool(action.name)
            validated = tool.param_model.model_validate(action.arguments)
            normalized_args = validated.model_dump(mode="json")
        except (ValueError, ValidationError) as exc:
            return Decision(
                type="reject",
                message=f"Invalid staged action for {action.name}: {exc}",
            )

        event_id = self._run_context.event_id
        if not event_id:
            return Decision(type="reject", message="Event identity is required")

        proposal = ActionProposal(
            organization_id=self._run_context.organization_id,
            user_id=self._run_context.user_id,
            event_id=event_id,
            origin_session_id=self._run_context.session_id,
            origin_run_id=self._run_context.run_id,
            tool=action.name,
            args=normalized_args,
            tool_schema_fingerprint=stable_fingerprint(
                tool.param_model.model_json_schema()
            ),
            reason=(action.description or "Agent requested an external action.")[:4000],
            impact=_proposal_impact(action.name),
            risk="high",
            routing="ask",
            status="pending",
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
            safe_default={"decision": "reject"},
        )
        trail = EventTrailEntry(
            organization_id=self._run_context.organization_id,
            user_id=self._run_context.user_id,
            event_id=event_id,
            actor_id=self._run_context.agent_id,
            kind="proposal_staged",
            summary=f"Staged {action.name} for approval",
            run_id=self._run_context.run_id,
            proposal_id=proposal.id,
            payload={
                "tool": action.name,
                "risk": proposal.risk,
                "routing": proposal.routing,
                "status": proposal.status,
                "transitions": ["proposed", "routed", "pending"],
            },
        )

        async with self._session_factory() as session:
            event = await session.get(Event, event_id)
            if (
                event is None
                or event.organization_id != self._run_context.organization_id
                or event.user_id != self._run_context.user_id
            ):
                return Decision(type="reject", message="Event is unavailable")
            session.add(proposal)
            session.add(trail)
            await session.commit()

        result = {"status": "staged", "proposal_id": proposal.id}
        if self._emit is not None:
            event_payload = _to_event(
                proposal.id,
                request,
                self._enrich,
            )
            event_payload.update(result)
            self._emit(event_payload)
        return Decision(type="defer", result=result)

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        decisions: List[Decision] = []
        for index in range(len(request.action_requests)):
            try:
                decision = await self._on_owner_loop(self._stage(request, index))
            except Exception:
                logger.exception("Could not stage Proposal")
                decision = Decision(
                    type="reject",
                    message="The external action could not be staged safely.",
                )
            decisions.append(decision)
        return ApprovalResponse(decisions=decisions)


class NonInteractiveDenyHandler(HITLHandler):
    """Compatibility safety floor for non-Event kernel integrations."""

    REASON = (
        "This action needs your approval, but it ran in the background where I "
        "can't ask. Start it in a live chat to review and approve it."
    )

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(
            decisions=[
                Decision(type="reject", message=self.REASON)
                for _ in request.action_requests
            ]
        )


def non_interactive_write_gate() -> tuple[NonInteractiveDenyHandler, HITLConfig]:
    """Compatibility gate for callers that cannot persist Aloy Proposals."""
    from .tools import GOOGLE_WRITE_TOOLS  # local: keep this module tool-agnostic

    return NonInteractiveDenyHandler(), build_write_hitl_config(GOOGLE_WRITE_TOOLS)


def proposal_write_gate(
    *,
    run_context: RunContext,
    tools_registry: Any,
    emit: Optional[Callable[[Dict[str, Any]], Any]] = None,
    enrich: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
    session_factory: Any = async_session,
    owner_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> tuple[ProposalStagingHandler, HITLConfig]:
    """Build the unified staging gate for interactive and background runs."""
    from .tools import GOOGLE_WRITE_TOOLS

    return (
        ProposalStagingHandler(
            run_context=run_context,
            tools_registry=tools_registry,
            emit=emit,
            enrich=enrich,
            session_factory=session_factory,
            owner_loop=owner_loop,
        ),
        build_write_hitl_config(GOOGLE_WRITE_TOOLS),
    )


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
