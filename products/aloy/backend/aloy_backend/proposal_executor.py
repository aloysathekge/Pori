"""Durable Proposal decisions and the non-agent execution commit rail.

The originating agent only stages serialized intent. This module is the sole
place that may turn an approved Proposal into an external consequence: it
re-authorizes against live tenant state, atomically claims one attempt,
executes the stored tool payload once, and persists receipt-backed truth.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import ValidationError
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from pori import (
    ReceiptStatus,
    ToolExecutionReceipt,
    register_all_tools,
    stable_fingerprint,
    tool_registry,
)
from pori.tools.registry import ToolExecutor, ToolRegistry

from .database import async_session
from .models import (
    ActionProposal,
    AgentConfig,
    Conversation,
    Event,
    EventTrailEntry,
    Organization,
    OrganizationMembership,
)
from .run_surface import resolve_run_surface
from .tenancy import ROLE_PERMISSIONS, OrganizationPolicy, Permission
from .tools import GOOGLE_WRITE_TOOLS, register_google_tools

logger = logging.getLogger("aloy_backend.proposals")

PROPOSAL_SAFE_TOOLS = frozenset(GOOGLE_WRITE_TOOLS)
STALE_EXECUTION_AFTER = timedelta(minutes=5)


class ProposalDecisionError(ValueError):
    """A safe, user-facing rejection of a Proposal decision."""

    def __init__(self, message: str, *, status_code: int = 409):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ProposalExecutionResult:
    proposal_id: str
    status: str
    claimed: bool
    error: str | None = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def proposal_tool_registry() -> ToolRegistry:
    """Resolve the current implementations for the explicit safe allowlist."""
    registry = tool_registry()
    register_all_tools(registry)
    register_google_tools(registry)
    return registry.filtered(
        include_tools=PROPOSAL_SAFE_TOOLS,
        protect_kernel=False,
    )


def _trail(
    proposal: ActionProposal,
    *,
    actor_id: str,
    kind: str,
    summary: str,
    payload: dict[str, Any],
    evidence_refs: list[dict[str, Any]] | None = None,
) -> EventTrailEntry:
    return EventTrailEntry(
        organization_id=proposal.organization_id,
        user_id=proposal.user_id,
        event_id=proposal.event_id,
        actor_id=actor_id,
        kind=kind,
        summary=summary,
        run_id=proposal.origin_run_id,
        proposal_id=proposal.id,
        evidence_refs=evidence_refs or [],
        payload=payload,
    )


async def _load_scoped_proposal(
    session: AsyncSession,
    *,
    event_id: str,
    proposal_id: str,
    organization_id: str,
    user_id: str,
) -> ActionProposal:
    result = await session.execute(
        select(ActionProposal).where(
            ActionProposal.id == proposal_id,
            ActionProposal.event_id == event_id,
            ActionProposal.organization_id == organization_id,
            ActionProposal.user_id == user_id,
        )
    )
    proposal = result.scalars().first()
    if proposal is None:
        raise ProposalDecisionError("Proposal not found", status_code=404)
    return proposal


async def _expire_loaded_proposal(
    session: AsyncSession,
    proposal: ActionProposal,
    *,
    now: datetime,
) -> bool:
    if (
        proposal.status != "pending"
        or proposal.expires_at is None
        or _as_utc(proposal.expires_at) > now
    ):
        return False
    result = await session.execute(
        update(ActionProposal)
        .execution_options(synchronize_session=False)
        .where(
            col(ActionProposal.id) == proposal.id,
            col(ActionProposal.status) == "pending",
            col(ActionProposal.expires_at).is_not(None),
            col(ActionProposal.expires_at) <= now,
        )
        .values(
            status="withdrawn",
            decided_by="system:expiry",
            decided_at=now,
            updated_at=now,
            error="Proposal expired; safe default rejected the action.",
        )
    )
    if result.rowcount != 1:  # type: ignore[attr-defined]
        return False
    session.add(
        _trail(
            proposal,
            actor_id="system:expiry",
            kind="proposal_decided",
            summary=f"Expired and rejected {proposal.tool}",
            payload={
                "decision": "reject",
                "reason": "expired",
                "safe_default": proposal.safe_default,
                "status": "withdrawn",
            },
        )
    )
    return True


async def decide_proposal(
    session: AsyncSession,
    *,
    event_id: str,
    proposal_id: str,
    organization_id: str,
    user_id: str,
    actor_id: str,
    decision: Literal["approve", "reject", "edit"],
    edited_action: dict[str, Any] | None = None,
    message: str | None = None,
    registry: ToolRegistry | None = None,
    now: datetime | None = None,
) -> ActionProposal:
    """Persist one tenant-scoped decision and its Trail entry atomically."""
    resolved_now = now or _utcnow()
    proposal = await _load_scoped_proposal(
        session,
        event_id=event_id,
        proposal_id=proposal_id,
        organization_id=organization_id,
        user_id=user_id,
    )
    if await _expire_loaded_proposal(session, proposal, now=resolved_now):
        await session.commit()
        raise ProposalDecisionError("Proposal has expired")
    if proposal.status != "pending":
        raise ProposalDecisionError(
            f"Proposal is {proposal.status}; only pending Proposals can be decided"
        )

    payload: dict[str, Any] = {"decision": decision}
    update_values: dict[str, Any] = {
        "decided_by": actor_id,
        "decided_at": resolved_now,
        "updated_at": resolved_now,
        "error": None,
    }
    if message:
        payload["message"] = message[:2000]
    if decision == "edit":
        if not isinstance(edited_action, dict):
            raise ProposalDecisionError("An edited action is required", status_code=422)
        edited_tool = edited_action.get("name") or edited_action.get("tool")
        if edited_tool != proposal.tool:
            raise ProposalDecisionError(
                "A Proposal edit cannot change the tool", status_code=422
            )
        edited_args = edited_action.get("args")
        if not isinstance(edited_args, dict):
            raise ProposalDecisionError(
                "Edited args must be an object", status_code=422
            )
        try:
            live_tool = (registry or proposal_tool_registry()).get_tool(proposal.tool)
            validated = live_tool.param_model.model_validate(edited_args)
        except (ValueError, ValidationError) as exc:
            raise ProposalDecisionError(
                f"Invalid edited action: {exc}", status_code=422
            ) from exc
        update_values["args"] = validated.model_dump(mode="json")
        update_values["tool_schema_fingerprint"] = stable_fingerprint(
            live_tool.param_model.model_json_schema()
        )
        # Editing changes serialized intent, so it remains pending and requires
        # a separate explicit approval.
        update_values.update(status="pending", risk="high", routing="ask")
        payload["status"] = "pending"
        payload["edited"] = True
        summary = f"Edited {proposal.tool}; approval is still required"
    elif decision == "approve":
        update_values["status"] = "approved"
        payload["status"] = "approved"
        summary = f"Approved {proposal.tool}"
    else:
        update_values["status"] = "withdrawn"
        payload["status"] = "withdrawn"
        summary = f"Rejected {proposal.tool}"

    result = await session.execute(
        update(ActionProposal)
        .execution_options(synchronize_session=False)
        .where(
            col(ActionProposal.id) == proposal.id,
            col(ActionProposal.status) == "pending",
        )
        .values(**update_values)
    )
    if result.rowcount != 1:  # type: ignore[attr-defined]
        await session.rollback()
        raise ProposalDecisionError("Proposal was already decided by another request")
    session.add(
        _trail(
            proposal,
            actor_id=actor_id,
            kind="proposal_decided",
            summary=summary,
            payload=payload,
        )
    )
    await session.commit()
    await session.refresh(proposal)
    return proposal


async def _mark_approved_failed(
    proposal: ActionProposal,
    error: str,
    *,
    session_factory: Any,
) -> ProposalExecutionResult:
    now = _utcnow()
    async with session_factory() as session:
        statement = (
            update(ActionProposal)
            .execution_options(synchronize_session=False)
            .where(
                col(ActionProposal.id) == proposal.id,
                col(ActionProposal.status) == "approved",
            )
            .values(status="failed", error=error[:4000], updated_at=now)
        )
        result = await session.execute(statement)
        if result.rowcount != 1:  # type: ignore[attr-defined]
            await session.rollback()
            return ProposalExecutionResult(proposal.id, proposal.status, False, error)
        session.add(
            _trail(
                proposal,
                actor_id="worker:proposal-executor",
                kind="proposal_failed",
                summary=f"Could not execute {proposal.tool}",
                payload={"status": "failed", "error": error[:1000]},
            )
        )
        await session.commit()
    return ProposalExecutionResult(proposal.id, "failed", False, error)


async def _authorize_and_resolve(
    proposal_id: str,
    *,
    session_factory: Any,
    registry: ToolRegistry | None,
) -> tuple[ActionProposal, ToolRegistry, dict[str, Any]] | ProposalExecutionResult:
    async with session_factory() as session:
        proposal = await session.get(ActionProposal, proposal_id)
        if proposal is None:
            return ProposalExecutionResult(proposal_id, "missing", False, "Not found")
        if proposal.status != "approved":
            return ProposalExecutionResult(proposal.id, proposal.status, False)

        membership_result = await session.execute(
            select(OrganizationMembership).where(
                OrganizationMembership.organization_id == proposal.organization_id,
                OrganizationMembership.user_id == proposal.user_id,
                OrganizationMembership.status == "active",
            )
        )
        membership = membership_result.scalars().first()
        organization = await session.get(Organization, proposal.organization_id)
        event = await session.get(Event, proposal.event_id)
        error: str | None = None
        policy: OrganizationPolicy | None = None
        if membership is None or organization is None:
            error = "Proposal owner no longer has organization access"
        elif Permission.RUN_CREATE not in ROLE_PERMISSIONS.get(
            membership.role, frozenset()
        ):
            error = "Proposal owner is no longer authorized to create external actions"
        elif (
            event is None
            or event.organization_id != proposal.organization_id
            or event.user_id != proposal.user_id
            or event.lifecycle != "active"
        ):
            error = "Proposal Event is unavailable or no longer active"
        elif proposal.tool not in PROPOSAL_SAFE_TOOLS:
            error = "Stored tool is not allowed on the Proposal executor"
        else:
            policy = OrganizationPolicy.model_validate(organization.policy or {})
            if policy.allowed_tools and proposal.tool not in policy.allowed_tools:
                error = "Tool is no longer allowed by organization policy"
            elif proposal.tool in policy.denied_tools:
                error = "Tool is denied by organization policy"
            elif (
                policy.allowed_capability_groups
                and "google" not in policy.allowed_capability_groups
            ):
                error = "Google capability is no longer allowed by organization policy"
            elif proposal.origin_session_id:
                conversation = await session.get(
                    Conversation, proposal.origin_session_id
                )
                if conversation is not None and (
                    conversation.organization_id != proposal.organization_id
                    or conversation.user_id != proposal.user_id
                    or conversation.event_id != proposal.event_id
                ):
                    error = "Proposal origin Session no longer matches its Event owner"
                elif conversation is not None and conversation.agent_config_id:
                    agent_config = await session.get(
                        AgentConfig, conversation.agent_config_id
                    )
                    if (
                        agent_config is None
                        or agent_config.organization_id != proposal.organization_id
                        or agent_config.user_id != proposal.user_id
                    ):
                        error = "Proposal Agent configuration is unavailable"
                    elif agent_config.tools and proposal.tool not in agent_config.tools:
                        error = "Tool is no longer allowed by the Proposal's Agent"

        if error is None and policy is not None:
            surface = await resolve_run_surface(
                session,
                organization_id=proposal.organization_id,
                user_id=proposal.user_id,
                policy=policy,
            )
            if proposal.tool in surface.denied_tools:
                error = "Tool capability or credentials are no longer available"
            else:
                live_registry = registry or proposal_tool_registry()
                try:
                    live_tool = live_registry.get_tool(proposal.tool)
                    if (
                        stable_fingerprint(live_tool.param_model.model_json_schema())
                        != proposal.tool_schema_fingerprint
                    ):
                        error = "Tool schema changed after this Proposal was staged"
                    else:
                        validated = live_tool.param_model.model_validate(proposal.args)
                        normalized = validated.model_dump(mode="json")
                        if normalized != proposal.args:
                            error = "Stored Proposal args are not normalized for the live schema"
                except (ValueError, ValidationError) as exc:
                    error = f"Stored Proposal payload is no longer executable: {exc}"
                if error is None:
                    context = surface.tool_context_extra
                    context.update(
                        {
                            "organization_id": proposal.organization_id,
                            "user_id": proposal.user_id,
                            "event_id": proposal.event_id,
                            "proposal_id": proposal.id,
                        }
                    )
                    return proposal, live_registry, context

    return await _mark_approved_failed(
        proposal,
        error or "Proposal authorization failed",
        session_factory=session_factory,
    )


async def _claim(
    proposal: ActionProposal,
    *,
    session_factory: Any,
) -> str | None:
    attempt_id = f"attempt_{uuid.uuid4().hex}"
    now = _utcnow()
    async with session_factory() as session:
        result = await session.execute(
            update(ActionProposal)
            .execution_options(synchronize_session=False)
            .where(
                col(ActionProposal.id) == proposal.id,
                col(ActionProposal.status) == "approved",
            )
            .values(
                status="executing",
                execution_attempt_id=attempt_id,
                error=None,
                updated_at=now,
            )
        )
        if result.rowcount != 1:  # type: ignore[attr-defined]
            await session.rollback()
            return None
        await session.commit()
    return attempt_id


async def _mark_indeterminate(
    proposal: ActionProposal,
    attempt_id: str,
    error: str,
    *,
    session_factory: Any,
) -> None:
    now = _utcnow()
    async with session_factory() as session:
        result = await session.execute(
            update(ActionProposal)
            .execution_options(synchronize_session=False)
            .where(
                col(ActionProposal.id) == proposal.id,
                col(ActionProposal.status) == "executing",
                col(ActionProposal.execution_attempt_id) == attempt_id,
            )
            .values(status="indeterminate", error=error[:4000], updated_at=now)
        )
        if result.rowcount != 1:  # type: ignore[attr-defined]
            await session.rollback()
            return
        session.add(
            _trail(
                proposal,
                actor_id="worker:proposal-executor",
                kind="proposal_indeterminate",
                summary=f"Outcome is uncertain for {proposal.tool}",
                payload={
                    "status": "indeterminate",
                    "execution_attempt_id": attempt_id,
                    "error": error[:1000],
                },
            )
        )
        await session.commit()


async def _finalize_execution(
    proposal: ActionProposal,
    attempt_id: str,
    *,
    outcome: dict[str, Any],
    receipt: ToolExecutionReceipt,
    provider_operation_id: str | None,
    session_factory: Any,
) -> str:
    success = receipt.status == ReceiptStatus.SUCCEEDED
    status = "committed" if success else "failed"
    error = None if success else receipt.error or "Tool execution failed"
    now = _utcnow()
    async with session_factory() as session:
        current = await session.get(ActionProposal, proposal.id)
        if (
            current is None
            or current.status != "executing"
            or current.execution_attempt_id != attempt_id
        ):
            raise RuntimeError("Proposal execution claim was lost before finalization")
        current.status = status
        current.receipt = receipt.model_dump(mode="json")
        current.provider_operation_id = provider_operation_id
        current.error = error
        current.updated_at = now
        session.add(current)
        evidence = [{"receipt_id": receipt.receipt_id}]
        if provider_operation_id:
            evidence.append({"provider_operation_id": provider_operation_id})
        session.add(
            _trail(
                current,
                actor_id="worker:proposal-executor",
                kind="proposal_committed" if success else "proposal_failed",
                summary=(
                    f"Committed {proposal.tool}"
                    if success
                    else f"Failed to execute {proposal.tool}"
                ),
                evidence_refs=evidence,
                payload={
                    "status": status,
                    "execution_attempt_id": attempt_id,
                    "provider_operation_id": provider_operation_id,
                    "outcome": outcome,
                    "error": error,
                },
            )
        )
        await session.commit()
    return status


async def execute_proposal(
    proposal_id: str,
    *,
    session_factory: Any = async_session,
    registry: ToolRegistry | None = None,
) -> ProposalExecutionResult:
    """Authorize, claim, execute once, and commit receipt-backed Proposal truth."""
    resolved = await _authorize_and_resolve(
        proposal_id,
        session_factory=session_factory,
        registry=registry,
    )
    if isinstance(resolved, ProposalExecutionResult):
        return resolved
    proposal, live_registry, context = resolved
    attempt_id = await _claim(proposal, session_factory=session_factory)
    if attempt_id is None:
        return ProposalExecutionResult(proposal.id, "not_claimed", False)
    context["execution_attempt_id"] = attempt_id

    started = _utcnow()
    try:
        outcome = await asyncio.to_thread(
            ToolExecutor(live_registry).execute_tool,
            proposal.tool,
            proposal.args,
            context,
        )
        result_payload = outcome.get("result")
        reported_error = (
            result_payload.get("error") if isinstance(result_payload, dict) else None
        )
        success = bool(outcome.get("success")) and not reported_error
        error = None if success else str(outcome.get("error") or reported_error)
        provider_operation_id = (
            str(result_payload.get("id"))
            if success and isinstance(result_payload, dict) and result_payload.get("id")
            else None
        )
        receipt = ToolExecutionReceipt(
            run_id=proposal.origin_run_id or f"proposal:{proposal.id}",
            tool_name=proposal.tool,
            status=ReceiptStatus.SUCCEEDED if success else ReceiptStatus.FAILED,
            backend="aloy-proposal-executor",
            parameters_fingerprint=stable_fingerprint(proposal.args),
            started_at=started,
            finished_at=_utcnow(),
            duration_seconds=max(0.0, (_utcnow() - started).total_seconds()),
            error=error,
            metadata={
                "proposal_id": proposal.id,
                "execution_attempt_id": attempt_id,
                "provider_operation_id": provider_operation_id,
                "result": result_payload,
            },
        )
        try:
            status = await _finalize_execution(
                proposal,
                attempt_id,
                outcome=outcome,
                receipt=receipt,
                provider_operation_id=provider_operation_id,
                session_factory=session_factory,
            )
        except Exception as exc:
            await _mark_indeterminate(
                proposal,
                attempt_id,
                f"Provider call finished but receipt persistence failed: {exc}",
                session_factory=session_factory,
            )
            return ProposalExecutionResult(proposal.id, "indeterminate", True, str(exc))
        return ProposalExecutionResult(proposal.id, status, True, error)
    except Exception as exc:
        # Once the provider call begins, an uncategorized exception is not safe
        # to retry: the provider may have accepted the request.
        logger.exception("Proposal %s execution outcome is uncertain", proposal.id)
        await _mark_indeterminate(
            proposal,
            attempt_id,
            f"Provider outcome is uncertain: {exc}",
            session_factory=session_factory,
        )
        return ProposalExecutionResult(proposal.id, "indeterminate", True, str(exc))


async def expire_due_proposals(
    *,
    session_factory: Any = async_session,
    now: datetime | None = None,
) -> int:
    """Apply the wedge's reject safe-default to unanswered expired Asks."""
    resolved_now = now or _utcnow()
    expired = 0
    async with session_factory() as session:
        proposals = (
            (
                await session.execute(
                    select(ActionProposal).where(
                        ActionProposal.status == "pending",
                        col(ActionProposal.expires_at).is_not(None),
                        col(ActionProposal.expires_at) <= resolved_now,
                    )
                )
            )
            .scalars()
            .all()
        )
        for proposal in proposals:
            if await _expire_loaded_proposal(session, proposal, now=resolved_now):
                expired += 1
        if expired:
            await session.commit()
    return expired


async def reconcile_stale_executions(
    *,
    session_factory: Any = async_session,
    now: datetime | None = None,
) -> int:
    """Never retry an abandoned external call; surface explicit uncertainty."""
    resolved_now = now or _utcnow()
    cutoff = resolved_now - STALE_EXECUTION_AFTER
    reconciled = 0
    async with session_factory() as session:
        proposals = (
            (
                await session.execute(
                    select(ActionProposal).where(
                        ActionProposal.status == "executing",
                        ActionProposal.updated_at <= cutoff,
                    )
                )
            )
            .scalars()
            .all()
        )
    for proposal in proposals:
        if not proposal.execution_attempt_id:
            continue
        await _mark_indeterminate(
            proposal,
            proposal.execution_attempt_id,
            "Executor claim became stale; provider outcome requires review.",
            session_factory=session_factory,
        )
        reconciled += 1
    return reconciled


async def execute_next_approved_proposal(
    *,
    session_factory: Any = async_session,
    registry: ToolRegistry | None = None,
) -> ProposalExecutionResult | None:
    """Worker fast-path: pick one approved Proposal; atomic claim decides winner."""
    async with session_factory() as session:
        proposal_id = (
            await session.execute(
                select(ActionProposal.id)
                .where(ActionProposal.status == "approved")
                .order_by(
                    col(ActionProposal.decided_at), col(ActionProposal.created_at)
                )
                .limit(1)
            )
        ).scalar_one_or_none()
    if proposal_id is None:
        return None
    return await execute_proposal(
        proposal_id,
        session_factory=session_factory,
        registry=registry,
    )


__all__ = [
    "PROPOSAL_SAFE_TOOLS",
    "ProposalDecisionError",
    "ProposalExecutionResult",
    "decide_proposal",
    "execute_next_approved_proposal",
    "execute_proposal",
    "expire_due_proposals",
    "proposal_tool_registry",
    "reconcile_stale_executions",
]
