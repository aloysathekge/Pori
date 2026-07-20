"""Read-only recovery of uncertain external Proposal outcomes.

Execution and reconciliation are intentionally separate rails. The executor
may perform a consequence exactly once; this module may only inspect a provider
through a tool's explicitly declared ``reconcile_fn`` and persist proven truth.
An unknown lookup remains indeterminate and can never authorize a repeat call.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import ValidationError
from sqlalchemy import or_, update
from sqlmodel import col, select

from pori import ReceiptStatus, ToolExecutionReceipt, stable_fingerprint
from pori.tools.registry import (
    ReconciliationStatus,
    ToolReconciliation,
    ToolRegistry,
)

from .database import async_session
from .models import (
    ActionProposal,
    Event,
    Organization,
    OrganizationMembership,
)
from .proposal_executor import (
    PROPOSAL_SAFE_TOOLS,
    ProposalExecutionResult,
    _trail,
    proposal_tool_registry,
)
from .run_surface import resolve_run_surface
from .surface_lifecycle import reconcile_surface_proposal
from .tenancy import OrganizationPolicy

RECONCILIATION_CLAIM_FOR = timedelta(minutes=2)
RECONCILIATION_RETRY_MIN = timedelta(seconds=30)
RECONCILIATION_RETRY_MAX = timedelta(hours=1)
RECONCILIATION_BATCH_SIZE = 10
RECONCILIATION_MAX_AUTOMATIC_ATTEMPTS = 8


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def _claim_reconciliation(
    proposal_id: str,
    execution_attempt_id: str,
    *,
    session_factory: Any,
    now: datetime,
) -> ActionProposal | None:
    """Lease one read-only provider inspection without changing consequence state."""
    async with session_factory() as session:
        result = await session.execute(
            update(ActionProposal)
            .execution_options(synchronize_session=False)
            .where(
                col(ActionProposal.id) == proposal_id,
                col(ActionProposal.status) == "indeterminate",
                col(ActionProposal.execution_attempt_id) == execution_attempt_id,
                or_(
                    col(ActionProposal.reconciliation_next_at).is_(None),
                    col(ActionProposal.reconciliation_next_at) <= now,
                ),
            )
            .values(
                reconciliation_attempts=ActionProposal.reconciliation_attempts + 1,
                reconciliation_checked_at=now,
                reconciliation_next_at=now + RECONCILIATION_CLAIM_FOR,
                updated_at=now,
            )
        )
        if result.rowcount != 1:  # type: ignore[attr-defined]
            await session.rollback()
            return None
        await session.commit()
        claimed = await session.get(ActionProposal, proposal_id)
        if claimed is not None:
            session.expunge(claimed)
        return claimed


def _reconciliation_delay(attempts: int) -> timedelta:
    exponent = min(7, max(0, attempts - 1))
    seconds = RECONCILIATION_RETRY_MIN.total_seconds() * (2**exponent)
    return timedelta(seconds=min(seconds, RECONCILIATION_RETRY_MAX.total_seconds()))


async def _schedule_reconciliation_retry(
    proposal: ActionProposal,
    error: str,
    *,
    session_factory: Any,
    now: datetime,
) -> None:
    """Retain uncertainty and back off; inspection never authorizes a resend."""
    exhausted = (
        proposal.reconciliation_attempts >= RECONCILIATION_MAX_AUTOMATIC_ATTEMPTS
    )
    message = (
        "Automatic provider reconciliation is paused after bounded attempts; "
        f"the outcome remains uncertain: {error}"
        if exhausted
        else "Provider outcome remains uncertain after read-only "
        f"reconciliation: {error}"
    )
    async with session_factory() as session:
        await session.execute(
            update(ActionProposal)
            .execution_options(synchronize_session=False)
            .where(
                col(ActionProposal.id) == proposal.id,
                col(ActionProposal.status) == "indeterminate",
                col(ActionProposal.execution_attempt_id)
                == proposal.execution_attempt_id,
            )
            .values(
                error=message[:4000],
                reconciliation_next_at=(
                    None
                    if exhausted
                    else now + _reconciliation_delay(proposal.reconciliation_attempts)
                ),
                updated_at=now,
            )
        )
        await session.commit()


async def _resolve_reconciliation_context(
    proposal: ActionProposal,
    *,
    session_factory: Any,
    registry: ToolRegistry,
) -> tuple[Any, dict[str, Any]] | str:
    """Resolve current read authority and credentials for provider inspection."""
    async with session_factory() as session:
        membership = (
            (
                await session.execute(
                    select(OrganizationMembership).where(
                        OrganizationMembership.organization_id
                        == proposal.organization_id,
                        OrganizationMembership.user_id == proposal.user_id,
                        OrganizationMembership.status == "active",
                    )
                )
            )
            .scalars()
            .first()
        )
        organization = await session.get(Organization, proposal.organization_id)
        event = await session.get(Event, proposal.event_id)
        if membership is None or organization is None:
            return "Proposal owner no longer has organization access"
        if event is None or (
            event.organization_id != proposal.organization_id
            or event.user_id != proposal.user_id
        ):
            return "Proposal Event is unavailable or no longer matches its owner"
        if proposal.tool not in PROPOSAL_SAFE_TOOLS:
            return "Stored tool is not allowed on the Proposal reconciler"
        try:
            live_tool = registry.get_tool(proposal.tool)
            if live_tool.reconcile_fn is None:
                return "The provider tool does not support read-only reconciliation"
            if (
                stable_fingerprint(live_tool.param_model.model_json_schema())
                != proposal.tool_schema_fingerprint
            ):
                return "Tool schema changed after this Proposal was staged"
            validated = live_tool.param_model.model_validate(proposal.args)
        except (ValueError, ValidationError) as exc:
            return f"Stored Proposal payload cannot be reconciled: {exc}"

        # Policy can revoke future writes without hiding truth about a past
        # attempt. The reconciler receives connection context only and invokes
        # the separately declared read-only callback, never the write function.
        surface = await resolve_run_surface(
            session,
            organization_id=proposal.organization_id,
            user_id=proposal.user_id,
            event_id=proposal.event_id,
            policy=OrganizationPolicy.model_validate(organization.policy or {}),
        )
        context = surface.tool_context_extra
        context.update(
            {
                "organization_id": proposal.organization_id,
                "user_id": proposal.user_id,
                "event_id": proposal.event_id,
                "proposal_id": proposal.id,
                "execution_attempt_id": proposal.execution_attempt_id,
            }
        )
        return validated, context


async def _finalize_reconciliation(
    proposal: ActionProposal,
    reconciliation: ToolReconciliation,
    *,
    session_factory: Any,
    started_at: datetime,
) -> bool:
    success = reconciliation.status == ReconciliationStatus.SUCCEEDED
    status = "committed" if success else "failed"
    now = _utcnow()
    error = None if success else reconciliation.error or "Provider rejected the action"
    result_payload = dict(reconciliation.result)
    receipt = ToolExecutionReceipt(
        run_id=proposal.origin_run_id or f"proposal:{proposal.id}",
        tool_name=proposal.tool,
        status=ReceiptStatus.SUCCEEDED if success else ReceiptStatus.FAILED,
        backend="aloy-proposal-reconciler",
        parameters_fingerprint=stable_fingerprint(proposal.args),
        started_at=started_at,
        finished_at=now,
        duration_seconds=max(0.0, (now - started_at).total_seconds()),
        error=error,
        metadata={
            "proposal_id": proposal.id,
            "execution_attempt_id": proposal.execution_attempt_id,
            "provider_operation_id": reconciliation.provider_operation_id,
            "result": result_payload,
            "reconciled": True,
            "reconciliation_attempt": proposal.reconciliation_attempts,
            "provider_evidence": [dict(item) for item in reconciliation.evidence],
        },
    )
    async with session_factory() as session:
        result = await session.execute(
            update(ActionProposal)
            .execution_options(synchronize_session=False)
            .where(
                col(ActionProposal.id) == proposal.id,
                col(ActionProposal.status) == "indeterminate",
                col(ActionProposal.execution_attempt_id)
                == proposal.execution_attempt_id,
            )
            .values(
                status=status,
                receipt=receipt.model_dump(mode="json"),
                provider_operation_id=reconciliation.provider_operation_id,
                error=error,
                reconciliation_next_at=None,
                updated_at=now,
            )
        )
        if result.rowcount != 1:  # type: ignore[attr-defined]
            await session.rollback()
            return False
        evidence_refs: list[dict[str, Any]] = [{"receipt_id": receipt.receipt_id}]
        if reconciliation.provider_operation_id:
            evidence_refs.append(
                {"provider_operation_id": reconciliation.provider_operation_id}
            )
        evidence_refs.extend(dict(item) for item in reconciliation.evidence)
        session.add(
            _trail(
                proposal,
                actor_id="worker:proposal-reconciler",
                kind="proposal_committed" if success else "proposal_failed",
                summary=(
                    f"Reconciled and committed {proposal.tool}"
                    if success
                    else f"Reconciled provider failure for {proposal.tool}"
                ),
                evidence_refs=evidence_refs,
                payload={
                    "status": status,
                    "execution_attempt_id": proposal.execution_attempt_id,
                    "provider_operation_id": reconciliation.provider_operation_id,
                    "outcome": result_payload,
                    "error": error,
                    "reconciled": True,
                    "reconciliation_attempt": proposal.reconciliation_attempts,
                },
            )
        )
        await reconcile_surface_proposal(
            session,
            proposal=proposal,
            proposal_status=status,
            error=error,
        )
        await session.commit()
    return True


async def reconcile_proposal_outcome(
    proposal_id: str,
    *,
    session_factory: Any = async_session,
    registry: ToolRegistry | None = None,
    now: datetime | None = None,
) -> ProposalExecutionResult:
    """Inspect one uncertain provider attempt and commit only proven truth."""
    resolved_now = now or _utcnow()
    live_registry = registry or proposal_tool_registry()
    async with session_factory() as session:
        proposal = await session.get(ActionProposal, proposal_id)
        if proposal is None:
            return ProposalExecutionResult(proposal_id, "missing", False, "Not found")
        if proposal.status != "indeterminate" or not proposal.execution_attempt_id:
            return ProposalExecutionResult(proposal.id, proposal.status, False)
        try:
            live_tool = live_registry.get_tool(proposal.tool)
        except ValueError as exc:
            return ProposalExecutionResult(
                proposal.id, "indeterminate", False, str(exc)
            )
        if live_tool.reconcile_fn is None:
            return ProposalExecutionResult(proposal.id, "indeterminate", False)

    claimed = await _claim_reconciliation(
        proposal.id,
        proposal.execution_attempt_id,
        session_factory=session_factory,
        now=resolved_now,
    )
    if claimed is None:
        return ProposalExecutionResult(proposal.id, "indeterminate", False)

    resolved = await _resolve_reconciliation_context(
        claimed,
        session_factory=session_factory,
        registry=live_registry,
    )
    if isinstance(resolved, str):
        await _schedule_reconciliation_retry(
            claimed,
            resolved,
            session_factory=session_factory,
            now=resolved_now,
        )
        return ProposalExecutionResult(claimed.id, "indeterminate", True, resolved)
    validated, context = resolved
    started_at = _utcnow()
    try:
        reconciliation = await asyncio.to_thread(
            live_tool.reconcile_fn,
            validated,
            context,
        )
    except Exception as exc:
        reconciliation = ToolReconciliation(
            status=ReconciliationStatus.UNKNOWN,
            error=f"Provider reconciliation raised an error: {exc}",
        )
    if not isinstance(reconciliation, ToolReconciliation):
        reconciliation = ToolReconciliation(
            status=ReconciliationStatus.UNKNOWN,
            error="Provider reconciler returned an invalid result",
        )
    if reconciliation.status == ReconciliationStatus.UNKNOWN:
        await _schedule_reconciliation_retry(
            claimed,
            reconciliation.error or "Provider returned no definitive outcome",
            session_factory=session_factory,
            now=resolved_now,
        )
        return ProposalExecutionResult(
            claimed.id,
            "indeterminate",
            True,
            reconciliation.error,
        )
    finalized = await _finalize_reconciliation(
        claimed,
        reconciliation,
        session_factory=session_factory,
        started_at=started_at,
    )
    status = (
        "committed"
        if reconciliation.status == ReconciliationStatus.SUCCEEDED
        else "failed"
    )
    return ProposalExecutionResult(
        claimed.id,
        status if finalized else "indeterminate",
        finalized,
        reconciliation.error,
    )


async def reconcile_indeterminate_proposals(
    *,
    session_factory: Any = async_session,
    registry: ToolRegistry | None = None,
    now: datetime | None = None,
    limit: int = RECONCILIATION_BATCH_SIZE,
) -> int:
    """Run bounded, leased, read-only checks for provider-supported tools."""
    resolved_now = now or _utcnow()
    live_registry = registry or proposal_tool_registry()
    supported_tools = tuple(
        sorted(
            name
            for name, info in live_registry.tools.items()
            if info.reconcile_fn is not None
        )
    )
    if not supported_tools or limit <= 0:
        return 0
    async with session_factory() as session:
        proposal_ids = (
            (
                await session.execute(
                    select(ActionProposal.id)
                    .where(
                        ActionProposal.status == "indeterminate",
                        col(ActionProposal.execution_attempt_id).is_not(None),
                        col(ActionProposal.tool).in_(supported_tools),
                        col(ActionProposal.reconciliation_next_at).is_not(None),
                        col(ActionProposal.reconciliation_next_at) <= resolved_now,
                    )
                    .order_by(
                        col(ActionProposal.reconciliation_next_at),
                        col(ActionProposal.updated_at),
                    )
                    .limit(min(limit, RECONCILIATION_BATCH_SIZE))
                )
            )
            .scalars()
            .all()
        )
    checked = 0
    for proposal_id in proposal_ids:
        result = await reconcile_proposal_outcome(
            proposal_id,
            session_factory=session_factory,
            registry=live_registry,
            now=resolved_now,
        )
        if result.claimed:
            checked += 1
    return checked


__all__ = [
    "reconcile_indeterminate_proposals",
    "reconcile_proposal_outcome",
]
