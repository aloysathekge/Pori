"""Durable, user-safe projection of kernel execution events.

Pori owns execution truth. Aloy deliberately projects only bounded public
milestones into this stream; raw arguments, tool results, token deltas, and
private reasoning remain outside the Work Story contract.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from pori.observability import (
    ACTIVITY_CHANGED,
    LLM_RETRY,
    PLAN_CHANGED,
    RUN_END,
    RUN_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    PoriEvent,
)

from .database import async_session
from .models import RunTimelineEvent

PUBLIC_EVENT_KINDS = frozenset(
    {
        "run_started",
        "activity_changed",
        "plan_changed",
        "action_started",
        "action_finished",
        "attention_required",
        "retrying",
        "run_finished",
        "run_failed",
    }
)


def project_pori_event(event: PoriEvent) -> tuple[str, dict[str, Any]] | None:
    """Convert one kernel event into a bounded public timeline milestone."""
    payload = event.payload or {}
    if event.type == RUN_START:
        return "run_started", {"status": "running"}
    if event.type == ACTIVITY_CHANGED:
        activity = str(payload.get("activity") or "").strip()
        return ("activity_changed", {"activity": activity}) if activity else None
    if event.type == PLAN_CHANGED:
        return "plan_changed", {
            "plan": list(payload.get("plan") or []),
            "summary": dict(payload.get("summary") or {}),
        }
    if event.type == TOOL_CALL_START:
        return "action_started", {
            "call_id": str(payload.get("call_id") or ""),
            "label": str(payload.get("label") or "Working"),
        }
    if event.type == TOOL_CALL_END:
        result: dict[str, Any] = {
            "call_id": str(payload.get("call_id") or ""),
            "label": str(payload.get("label") or "Finished an action"),
            "success": bool(payload.get("success")),
        }
        if isinstance(payload.get("duration_seconds"), (int, float)):
            result["duration_seconds"] = float(payload["duration_seconds"])
        return "action_finished", result
    if event.type in {"clarification_request", "approval_request"}:
        return "attention_required", {
            "request_id": str(payload.get("id") or ""),
            "request_kind": (
                "approval" if event.type == "approval_request" else "clarification"
            ),
            "description": str(
                payload.get("description") or payload.get("question") or ""
            )[:1000],
        }
    if event.type == LLM_RETRY:
        return "retrying", {"status": "retrying"}
    if event.type == RUN_END:
        return "run_finished", {
            "completed": bool(payload.get("completed")),
            "steps": int(payload.get("steps") or event.step or 0),
        }
    return None


class RunTimelineRecorder:
    """Append projected events with a monotonic per-Run sequence."""

    def __init__(
        self,
        *,
        organization_id: str,
        user_id: str,
        event_id: str,
        conversation_id: str | None,
        run_id: str,
        session_factory: Callable[[], AsyncSession] = async_session,
    ) -> None:
        self.organization_id = organization_id
        self.user_id = user_id
        self.event_id = event_id
        self.conversation_id = conversation_id
        self.run_id = run_id
        self._session_factory = session_factory
        self._sequence: int | None = None
        self._lock = asyncio.Lock()

    async def record(self, event: PoriEvent) -> RunTimelineEvent | None:
        projected = project_pori_event(event)
        if projected is None:
            return None
        kind, payload = projected
        return await self.append(kind, payload)

    async def append(
        self, kind: str, public_payload: dict[str, Any]
    ) -> RunTimelineEvent:
        if kind not in PUBLIC_EVENT_KINDS:
            raise ValueError(f"Unsupported Run timeline kind: {kind}")
        async with self._lock:
            async with self._session_factory() as session:
                if self._sequence is None:
                    current = (
                        await session.execute(
                            select(
                                func.coalesce(func.max(RunTimelineEvent.sequence), 0)
                            ).where(RunTimelineEvent.run_id == self.run_id)
                        )
                    ).scalar_one()
                    self._sequence = int(current)
                self._sequence += 1
                row = RunTimelineEvent(
                    organization_id=self.organization_id,
                    user_id=self.user_id,
                    event_id=self.event_id,
                    conversation_id=self.conversation_id,
                    run_id=self.run_id,
                    sequence=self._sequence,
                    kind=kind,
                    public_payload=public_payload,
                )
                session.add(row)
                await session.commit()
                await session.refresh(row)
                return row


__all__ = [
    "PUBLIC_EVENT_KINDS",
    "RunTimelineRecorder",
    "project_pori_event",
]
