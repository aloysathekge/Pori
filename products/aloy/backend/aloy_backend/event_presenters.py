"""Typed JSON presenters for Event Surfaces and Today.

These functions deliberately project database-owned truth. The app never
renders an external consequence from agent prose: pending reality comes from
Proposal rows and committed reality carries receipt/provider evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .models import ActionProposal, Event, EventTrailEntry, StoredFile, Task


def _utc_payload(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def event_payload(event: Event) -> dict[str, Any]:
    return {
        "id": event.id,
        "type": event.type,
        "title": event.title,
        "lifecycle": event.lifecycle,
        "phase": event.phase,
        "summary": event.summary,
        "is_life": event.is_life,
        "conversation_id": event.primary_conversation_id,
        "origin_conversation_id": (event.metadata_ or {}).get("origin_conversation_id"),
        "created_at": event.created_at,
        "updated_at": event.updated_at,
    }


def task_payload(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "event_id": task.event_id,
        "origin_conversation_id": task.origin_conversation_id,
        "title": task.title,
        "status": task.status,
        "instructions": task.instructions,
        "definition_of_done": task.definition_of_done,
        "priority": task.priority,
        "due_at": _utc_payload(task.due_at),
        "execution_mode": task.execution_mode,
        "assigned_agent_id": task.assigned_agent_id,
        "current_run_id": task.current_run_id,
        "result_summary": task.result_summary,
        "blocker": task.blocker,
        "budget_policy": task.budget_policy,
        "order": task.order,
        "created_by": task.created_by,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


def proposal_payload(proposal: ActionProposal) -> dict[str, Any]:
    return {
        "id": proposal.id,
        "event_id": proposal.event_id,
        "tool": proposal.tool,
        "args": proposal.args,
        "reason": proposal.reason,
        "impact": proposal.impact,
        "risk": proposal.risk,
        "routing": proposal.routing,
        "status": proposal.status,
        "expires_at": proposal.expires_at,
        "decided_at": proposal.decided_at,
        "provider_operation_id": proposal.provider_operation_id,
        "receipt": proposal.receipt,
        "error": proposal.error,
        "created_at": proposal.created_at,
        "updated_at": proposal.updated_at,
    }


def trail_payload(entry: EventTrailEntry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "kind": entry.kind,
        "summary": entry.summary,
        "actor_id": entry.actor_id,
        "run_id": entry.run_id,
        "proposal_id": entry.proposal_id,
        "task_id": entry.task_id,
        "evidence_refs": entry.evidence_refs,
        "payload": entry.payload,
        "created_at": entry.created_at,
    }


def file_payload(file: StoredFile) -> dict[str, Any]:
    return {
        "id": file.id,
        "name": file.name,
        "kind": file.kind,
        "content_type": file.content_type,
        "size_bytes": file.size_bytes,
        "origin_session_id": file.origin_session_id,
        "origin_run_id": file.run_id,
        "created_at": file.created_at,
    }


__all__ = [
    "event_payload",
    "file_payload",
    "proposal_payload",
    "task_payload",
    "trail_payload",
]
