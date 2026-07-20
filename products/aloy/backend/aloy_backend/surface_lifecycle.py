"""Exactly-once lifecycle reconciliation for model-authored Surface requests.

Generated UI may request reasoning or stage an external action, but only the
trusted worker and Proposal executor can complete those requests.  This module
keeps the originating ``SurfaceInteraction``, permanent Event Conversation,
and semantic Trail in one transaction with each trusted state transition.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from .models import (
    ActionProposal,
    Conversation,
    EventTrailEntry,
    Message,
    Run,
    SurfaceInteraction,
)

RUN_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
ACTION_TERMINAL_STATUSES = frozenset(
    {"committed", "rejected", "failed", "indeterminate"}
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _action_label(tool: str) -> str:
    return tool.replace("_", " ").strip().capitalize() or "External action"


async def surface_interaction_for_run(
    session: AsyncSession, run_id: str
) -> SurfaceInteraction | None:
    return (
        (
            await session.execute(
                select(SurfaceInteraction).where(
                    SurfaceInteraction.handling_run_id == run_id
                )
            )
        )
        .scalars()
        .first()
    )


async def _interaction_for_proposal(
    session: AsyncSession, proposal_id: str
) -> SurfaceInteraction | None:
    return (
        (
            await session.execute(
                select(SurfaceInteraction).where(
                    SurfaceInteraction.proposal_id == proposal_id
                )
            )
        )
        .scalars()
        .first()
    )


async def _touch_conversation(
    session: AsyncSession, interaction: SurfaceInteraction, now: datetime
) -> None:
    if not interaction.conversation_id:
        return
    conversation = await session.get(Conversation, interaction.conversation_id)
    if (
        conversation is None
        or conversation.event_id != interaction.event_id
        or conversation.organization_id != interaction.organization_id
        or conversation.user_id != interaction.user_id
    ):
        return
    conversation.updated_at = now
    session.add(conversation)


async def stage_surface_action_message(
    session: AsyncSession,
    *,
    interaction: SurfaceInteraction,
    proposal: ActionProposal,
) -> Message | None:
    """Stage one host-owned request card with the interaction and Proposal."""
    if interaction.request_message_id or not interaction.conversation_id:
        return None
    message = Message(
        conversation_id=interaction.conversation_id,
        role="assistant",
        content=(
            f"Requested **{_action_label(proposal.tool)}** from this Surface. "
            "Review the pending approval before anything is performed."
        ),
        metadata_={
            "kind": "surface_action_lifecycle",
            "phase": "request",
            "status": "waiting_approval",
            "surface_interaction_id": interaction.id,
            "proposal_id": proposal.id,
            "tool": proposal.tool,
        },
    )
    session.add(message)
    interaction.request_message_id = message.id
    now = _utcnow()
    interaction.updated_at = now
    session.add(interaction)
    await _touch_conversation(session, interaction, now)
    return message


async def mark_surface_run_started(session: AsyncSession, *, run: Run) -> bool:
    """Move a queued Surface reasoning request to running exactly once."""
    interaction = await surface_interaction_for_run(session, run.id)
    if interaction is None or interaction.status != "queued":
        return False
    now = _utcnow()
    interaction.status = "running"
    interaction.updated_at = now
    interaction.result = {
        **(interaction.result or {}),
        "run_id": run.id,
        "run_status": "running",
    }
    session.add(interaction)
    session.add(
        EventTrailEntry(
            organization_id=interaction.organization_id,
            user_id=interaction.user_id,
            event_id=interaction.event_id,
            actor_id="worker:surface-reasoning",
            kind="surface_reasoning_started",
            summary="Started a reasoning request from the Event Surface",
            run_id=run.id,
            evidence_refs=[{"surface_interaction_id": interaction.id}],
            payload={"interaction_id": interaction.id, "status": "running"},
        )
    )
    return True


def _run_interaction_status(run: Run) -> str | None:
    if run.status == "cancelled":
        return "cancelled"
    if run.status == "failed" or (run.status == "completed" and run.success is False):
        return "failed"
    if run.status == "completed":
        return "completed"
    return None


async def reconcile_surface_run(
    session: AsyncSession,
    *,
    run: Run,
    outcome_message: Message | None = None,
    error: str | None = None,
) -> SurfaceInteraction | None:
    """Reconcile one trusted terminal Run into its Surface request."""
    status = _run_interaction_status(run)
    if status is None:
        return None
    interaction = await surface_interaction_for_run(session, run.id)
    if interaction is None:
        return None
    if interaction.status in RUN_TERMINAL_STATUSES:
        return interaction

    now = _utcnow()
    interaction.status = status
    interaction.updated_at = now
    interaction.error = error[:4000] if error is not None else None
    interaction.result = {
        **(interaction.result or {}),
        "run_id": run.id,
        "run_status": run.status,
        "success": bool(run.success),
        "steps_taken": run.steps_taken,
    }

    message = outcome_message
    if message is None and interaction.conversation_id:
        if status == "cancelled":
            content = "The reasoning request from this Surface was stopped."
        else:
            detail = f" {error[:500]}" if error else ""
            content = (
                f"The reasoning request from this Surface could not finish.{detail}"
            )
        message = Message(
            conversation_id=interaction.conversation_id,
            role="assistant",
            content=content,
            metadata_={
                "kind": "surface_reasoning_result",
                "status": status,
                "surface_interaction_id": interaction.id,
                "run_id": run.id,
            },
        )
        session.add(message)
    elif message is not None:
        message.metadata_ = {
            **(message.metadata_ or {}),
            "kind": "surface_reasoning_result",
            "status": status,
            "surface_interaction_id": interaction.id,
            "run_id": run.id,
        }
        session.add(message)

    if message is not None:
        interaction.outcome_message_id = message.id
        interaction.result = {
            **interaction.result,
            "message_id": message.id,
            "conversation_id": interaction.conversation_id,
        }

    trail_kind = {
        "completed": "surface_reasoning_completed",
        "failed": "surface_reasoning_failed",
        "cancelled": "surface_reasoning_cancelled",
    }[status]
    summary = {
        "completed": "Completed a reasoning request from the Event Surface",
        "failed": "A reasoning request from the Event Surface failed",
        "cancelled": "Stopped a reasoning request from the Event Surface",
    }[status]
    session.add(interaction)
    session.add(
        EventTrailEntry(
            organization_id=interaction.organization_id,
            user_id=interaction.user_id,
            event_id=interaction.event_id,
            actor_id=run.agent_id or "agent",
            kind=trail_kind,
            summary=summary,
            run_id=run.id,
            evidence_refs=[{"surface_interaction_id": interaction.id}],
            payload={
                "interaction_id": interaction.id,
                "status": status,
                "message_id": message.id if message is not None else None,
            },
        )
    )
    await _touch_conversation(session, interaction, now)
    return interaction


def _proposal_interaction_status(proposal_status: str) -> str:
    return {
        "pending": "waiting_approval",
        "approved": "approved",
        "executing": "executing",
        "withdrawn": "rejected",
        "committed": "committed",
        "failed": "failed",
        "indeterminate": "indeterminate",
    }.get(proposal_status, proposal_status)


def _action_outcome_content(
    proposal: ActionProposal, *, status: str, error: str | None
) -> str:
    label = _action_label(proposal.tool)
    if status == "committed":
        return f"**{label}** completed. Aloy recorded a verified receipt."
    if status == "rejected":
        return f"**{label}** was not performed because the request was rejected."
    if status == "indeterminate":
        return (
            f"Aloy cannot yet confirm whether **{label}** completed. "
            "The action will not be retried blindly and requires reconciliation."
        )
    detail = f" {error[:500]}" if error else ""
    return f"**{label}** failed and was not marked complete.{detail}"


async def reconcile_surface_proposal(
    session: AsyncSession,
    *,
    proposal: ActionProposal,
    proposal_status: str | None = None,
    error: str | None = None,
) -> SurfaceInteraction | None:
    """Mirror trusted Proposal state and terminal evidence into its request."""
    interaction = await _interaction_for_proposal(session, proposal.id)
    if interaction is None:
        return None
    resolved_proposal_status = proposal_status or proposal.status
    status = _proposal_interaction_status(resolved_proposal_status)
    if interaction.status == status and (
        status not in ACTION_TERMINAL_STATUSES or interaction.outcome_message_id
    ):
        return interaction
    if interaction.status in ACTION_TERMINAL_STATUSES:
        return interaction

    previous = interaction.status
    now = _utcnow()
    interaction.status = status
    interaction.updated_at = now
    interaction.error = error or proposal.error or None
    interaction.result = {
        **(interaction.result or {}),
        "proposal_id": proposal.id,
        "proposal_status": resolved_proposal_status,
        "provider_operation_id": proposal.provider_operation_id,
        "receipt": proposal.receipt,
    }

    if status == "executing" and previous != "executing":
        session.add(
            EventTrailEntry(
                organization_id=interaction.organization_id,
                user_id=interaction.user_id,
                event_id=interaction.event_id,
                actor_id="worker:proposal-executor",
                kind="surface_action_started",
                summary=f"Started {_action_label(proposal.tool)} from the Event Surface",
                proposal_id=proposal.id,
                evidence_refs=[{"surface_interaction_id": interaction.id}],
                payload={"interaction_id": interaction.id, "status": "executing"},
            )
        )

    if status in ACTION_TERMINAL_STATUSES and not interaction.outcome_message_id:
        if interaction.conversation_id:
            message = Message(
                conversation_id=interaction.conversation_id,
                role="assistant",
                content=_action_outcome_content(
                    proposal,
                    status=status,
                    error=interaction.error,
                ),
                metadata_={
                    "kind": "surface_action_lifecycle",
                    "phase": "outcome",
                    "status": status,
                    "surface_interaction_id": interaction.id,
                    "proposal_id": proposal.id,
                    "tool": proposal.tool,
                    "has_receipt": proposal.receipt is not None,
                },
            )
            session.add(message)
            interaction.outcome_message_id = message.id
            interaction.result = {**interaction.result, "message_id": message.id}
            await _touch_conversation(session, interaction, now)

    session.add(interaction)
    return interaction


__all__ = [
    "mark_surface_run_started",
    "reconcile_surface_proposal",
    "reconcile_surface_run",
    "stage_surface_action_message",
    "surface_interaction_for_run",
]
