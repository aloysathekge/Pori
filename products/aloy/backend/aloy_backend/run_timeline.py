"""Durable, user-safe projection of kernel execution events.

Pori owns execution truth. Aloy deliberately projects only bounded public
milestones into this stream; raw arguments, tool results, token deltas, and
private reasoning remain outside the Work Story contract.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from pori import (
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
from .models import Run, RunTimelineCursor, RunTimelineEvent

logger = logging.getLogger("aloy_backend.run_timeline")

TIMELINE_SCHEMA_VERSION = 1
MAX_PUBLIC_PAYLOAD_BYTES = 64 * 1024
MAX_APPEND_ATTEMPTS = 5
MAX_TIMELINE_EVENTS_PER_RUN = 50_000

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
        "run_cancelled",
    }
)


class _PublicPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunStartedPayload(_PublicPayload):
    status: Literal["running"]


class ActivityChangedPayload(_PublicPayload):
    activity: str = Field(min_length=1, max_length=1000)


class PublicPlanItem(_PublicPayload):
    id: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1, max_length=2000)
    status: Literal["pending", "in_progress", "completed", "cancelled"]


class PlanChangedPayload(_PublicPayload):
    plan: list[PublicPlanItem] = Field(max_length=200)
    summary: dict[str, int] = Field(default_factory=dict)


class ActionStartedPayload(_PublicPayload):
    call_id: str = Field(max_length=300)
    label: str = Field(min_length=1, max_length=500)


class ActionFinishedPayload(ActionStartedPayload):
    success: bool
    duration_seconds: float | None = Field(default=None, ge=0, le=86_400)


class AttentionRequiredPayload(_PublicPayload):
    request_id: str = Field(max_length=300)
    request_kind: Literal["approval", "clarification"]
    description: str = Field(max_length=1000)


class RetryingPayload(_PublicPayload):
    status: Literal["retrying"]


class RunFinishedPayload(_PublicPayload):
    completed: bool
    steps: int = Field(ge=0, le=1_000_000)


class RunFailedPayload(_PublicPayload):
    status: Literal["retrying", "failed"] | None = None
    reason: str | None = Field(default=None, max_length=300)
    message: str | None = Field(default=None, max_length=1000)


class RunCancelledPayload(_PublicPayload):
    reason: str | None = Field(default=None, max_length=300)


_PAYLOAD_MODELS: dict[str, type[_PublicPayload]] = {
    "run_started": RunStartedPayload,
    "activity_changed": ActivityChangedPayload,
    "plan_changed": PlanChangedPayload,
    "action_started": ActionStartedPayload,
    "action_finished": ActionFinishedPayload,
    "attention_required": AttentionRequiredPayload,
    "retrying": RetryingPayload,
    "run_finished": RunFinishedPayload,
    "run_failed": RunFailedPayload,
    "run_cancelled": RunCancelledPayload,
}


class TimelineNotifier(Protocol):
    """Wake live readers; durable cursor replay remains the source of truth."""

    def publish(self, run_id: str) -> None: ...

    async def wait(self, run_id: str, timeout: float) -> None: ...


class LocalTimelineNotifier:
    """Low-latency same-process wakeups with polling-compatible timeouts.

    Hosted multi-process deployments can replace this contract with Postgres
    LISTEN/NOTIFY or a broker. Missing a wakeup is harmless because readers
    always query from their durable sequence cursor after the timeout.
    """

    def __init__(self, *, max_tracked_runs: int = 2048) -> None:
        self.max_tracked_runs = max_tracked_runs
        self._events: OrderedDict[str, asyncio.Event] = OrderedDict()

    def _event(self, run_id: str) -> asyncio.Event:
        event = self._events.get(run_id)
        if event is None:
            event = asyncio.Event()
            self._events[run_id] = event
            while len(self._events) > self.max_tracked_runs:
                self._events.popitem(last=False)
        else:
            self._events.move_to_end(run_id)
        return event

    def publish(self, run_id: str) -> None:
        self._event(run_id).set()

    async def wait(self, run_id: str, timeout: float) -> None:
        event = self._event(run_id)
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return
        finally:
            event.clear()


timeline_notifier: TimelineNotifier = LocalTimelineNotifier()


def validate_public_payload(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical version-1 payload or reject unsafe drift."""
    model = _PAYLOAD_MODELS.get(kind)
    if model is None:
        raise ValueError(f"Unsupported Run timeline kind: {kind}")
    validated = model.model_validate(payload).model_dump(mode="json", exclude_none=True)
    encoded = json.dumps(
        validated, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if len(encoded) > MAX_PUBLIC_PAYLOAD_BYTES:
        raise ValueError("Run timeline payload exceeds the public size limit")
    return validated


def _payload_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_event_key(
    event: PoriEvent, kind: str, public_payload: dict[str, Any]
) -> str:
    """Build a stable replay identity without retaining private input."""
    call_id = public_payload.get("call_id")
    if isinstance(call_id, str) and call_id:
        return f"pori:{event.type}:{call_id}"
    request_id = public_payload.get("request_id")
    if isinstance(request_id, str) and request_id:
        return f"pori:{event.type}:{request_id}"
    if event.type in {RUN_START, RUN_END}:
        return f"pori:{event.type}"
    fingerprint = _payload_digest(
        {
            "event_type": event.type,
            "step": event.step,
            "kind": kind,
            "payload": public_payload,
        }
    )
    return f"pori:{event.type}:{event.step}:{fingerprint}"


def project_pori_event(event: PoriEvent) -> tuple[str, dict[str, Any]] | None:
    """Convert one kernel event into a bounded public timeline milestone."""
    payload = event.payload or {}
    projected: tuple[str, dict[str, Any]] | None = None
    if event.type == RUN_START:
        projected = "run_started", {"status": "running"}
    elif event.type == ACTIVITY_CHANGED:
        activity = str(payload.get("activity") or "").strip()
        projected = ("activity_changed", {"activity": activity}) if activity else None
    elif event.type == PLAN_CHANGED:
        projected = "plan_changed", {
            "plan": list(payload.get("plan") or []),
            "summary": dict(payload.get("summary") or {}),
        }
    elif event.type == TOOL_CALL_START:
        projected = "action_started", {
            "call_id": str(payload.get("call_id") or ""),
            "label": str(payload.get("label") or "Working"),
        }
    elif event.type == TOOL_CALL_END:
        result: dict[str, Any] = {
            "call_id": str(payload.get("call_id") or ""),
            "label": str(payload.get("label") or "Finished an action"),
            "success": bool(payload.get("success")),
        }
        if isinstance(payload.get("duration_seconds"), (int, float)):
            result["duration_seconds"] = float(payload["duration_seconds"])
        projected = "action_finished", result
    elif event.type in {"clarification_request", "approval_request"}:
        projected = "attention_required", {
            "request_id": str(payload.get("id") or ""),
            "request_kind": (
                "approval" if event.type == "approval_request" else "clarification"
            ),
            "description": str(
                payload.get("description") or payload.get("question") or ""
            )[:1000],
        }
    elif event.type == LLM_RETRY:
        projected = "retrying", {"status": "retrying"}
    elif event.type == RUN_END:
        projected = "run_finished", {
            "completed": bool(payload.get("completed")),
            "steps": int(payload.get("steps") or event.step or 0),
        }
    if projected is None:
        return None
    kind, public_payload = projected
    return kind, validate_public_payload(kind, public_payload)


class RunTimelineRecorder:
    """Idempotently append projected events with an atomic per-Run sequence."""

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
        self._lock = asyncio.Lock()

    async def record(self, event: PoriEvent) -> RunTimelineEvent | None:
        projected = project_pori_event(event)
        if projected is None:
            return None
        kind, payload = projected
        return await self.append(
            kind,
            payload,
            source_event_key=source_event_key(event, kind, payload),
        )

    async def _existing(
        self, session: AsyncSession, source_key: str
    ) -> RunTimelineEvent | None:
        return (
            (
                await session.execute(
                    select(RunTimelineEvent).where(
                        RunTimelineEvent.run_id == self.run_id,
                        RunTimelineEvent.source_event_key == source_key,
                    )
                )
            )
            .scalars()
            .first()
        )

    async def _allocate_sequence(self, session: AsyncSession) -> int:
        cursor_table: Any = getattr(RunTimelineCursor, "__table__")
        allocated = (
            await session.execute(
                update(cursor_table)
                .where(
                    cursor_table.c.run_id == self.run_id,
                    cursor_table.c.last_sequence < MAX_TIMELINE_EVENTS_PER_RUN,
                )
                .values(last_sequence=cursor_table.c.last_sequence + 1)
                .returning(cursor_table.c.last_sequence)
            )
        ).scalar_one_or_none()
        if allocated is not None:
            return int(allocated)
        existing_cursor = await session.get(RunTimelineCursor, self.run_id)
        if existing_cursor is not None:
            raise ValueError("Run timeline reached its durable event limit")
        cursor = RunTimelineCursor(run_id=self.run_id, last_sequence=1)
        session.add(cursor)
        await session.flush()
        return 1

    async def append(
        self,
        kind: str,
        public_payload: dict[str, Any],
        *,
        source_event_key: str | None = None,
    ) -> RunTimelineEvent:
        validated = validate_public_payload(kind, public_payload)
        source_key = source_event_key or (f"host:{kind}:{_payload_digest(validated)}")
        async with self._lock:
            for attempt in range(MAX_APPEND_ATTEMPTS):
                async with self._session_factory() as session:
                    existing = await self._existing(session, source_key)
                    if existing is not None:
                        return existing
                    try:
                        sequence = await self._allocate_sequence(session)
                        row = RunTimelineEvent(
                            organization_id=self.organization_id,
                            user_id=self.user_id,
                            event_id=self.event_id,
                            conversation_id=self.conversation_id,
                            run_id=self.run_id,
                            sequence=sequence,
                            kind=kind,
                            schema_version=TIMELINE_SCHEMA_VERSION,
                            source_event_key=source_key,
                            public_payload=validated,
                        )
                        session.add(row)
                        await session.commit()
                        await session.refresh(row)
                        timeline_notifier.publish(self.run_id)
                        return row
                    except (IntegrityError, OperationalError):
                        await session.rollback()
                        existing = await self._existing(session, source_key)
                        if existing is not None:
                            return existing
                if attempt + 1 < MAX_APPEND_ATTEMPTS:
                    await asyncio.sleep(0.01 * (2**attempt))
            raise RuntimeError(
                f"Could not append Run timeline event after {MAX_APPEND_ATTEMPTS} attempts"
            )

    async def append_safely(
        self,
        kind: str,
        public_payload: dict[str, Any],
        *,
        source_event_key: str | None = None,
    ) -> RunTimelineEvent | None:
        """Persist observability without making it an execution dependency."""
        try:
            return await self.append(
                kind, public_payload, source_event_key=source_event_key
            )
        except Exception:
            logger.warning("Run timeline event could not be persisted", exc_info=True)
            return None


async def reconcile_terminal_run_timeline(
    run_id: str,
    *,
    session_factory: Callable[[], AsyncSession] = async_session,
) -> RunTimelineEvent | None:
    """Repair a missing terminal milestone from canonical committed Run truth."""
    async with session_factory() as session:
        run = await session.get(Run, run_id)
        if run is None or run.status not in {"completed", "failed", "cancelled"}:
            return None
        terminal_kind = (
            "run_cancelled"
            if run.status == "cancelled"
            else (
                "run_finished"
                if run.status == "completed" and run.success
                else "run_failed"
            )
        )
        existing = (
            (
                await session.execute(
                    select(RunTimelineEvent).where(
                        RunTimelineEvent.run_id == run.id,
                        RunTimelineEvent.kind == terminal_kind,
                    )
                )
            )
            .scalars()
            .first()
        )
        if existing is not None and (
            terminal_kind != "run_finished"
            or existing.public_payload.get("completed") is True
        ):
            return existing

    recorder = RunTimelineRecorder(
        organization_id=run.organization_id,
        user_id=run.user_id,
        event_id=run.event_id,
        conversation_id=run.conversation_id,
        run_id=run.id,
        session_factory=session_factory,
    )
    if terminal_kind == "run_finished":
        payload: dict[str, Any] = {
            "completed": True,
            "steps": max(0, run.steps_taken),
        }
    elif terminal_kind == "run_cancelled":
        payload = {"reason": "cancelled"}
    else:
        payload = {"status": "failed", "reason": "terminal_reconciliation"}
    return await recorder.append_safely(
        terminal_kind,
        payload,
        source_event_key=f"terminal:{terminal_kind}",
    )


class AsyncRunTimelineSink:
    """Order-preserving bridge from Pori's synchronous event callback."""

    def __init__(self, recorder: RunTimelineRecorder) -> None:
        self.recorder = recorder
        self._queue: asyncio.Queue[PoriEvent | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._drain())

    def emit(self, event: PoriEvent) -> None:
        self._queue.put_nowait(event)

    async def close(self) -> None:
        if self._worker is None:
            return
        await self._queue.put(None)
        await self._worker
        self._worker = None

    async def _drain(self) -> None:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            projected = project_pori_event(event)
            if projected is not None:
                kind, payload = projected
                await self.recorder.append_safely(
                    kind,
                    payload,
                    source_event_key=source_event_key(event, kind, payload),
                )


__all__ = [
    "MAX_PUBLIC_PAYLOAD_BYTES",
    "MAX_TIMELINE_EVENTS_PER_RUN",
    "PUBLIC_EVENT_KINDS",
    "TIMELINE_SCHEMA_VERSION",
    "AsyncRunTimelineSink",
    "LocalTimelineNotifier",
    "RunTimelineRecorder",
    "project_pori_event",
    "reconcile_terminal_run_timeline",
    "source_event_key",
    "timeline_notifier",
    "validate_public_payload",
]
