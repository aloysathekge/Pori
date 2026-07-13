"""One canonical run outcome, one persist path.

The streaming (SSE) and non-streaming request handlers both produce ONE
``RunOutcome`` from the agent's result, and both persist it through the single
``persist_run_outcome`` finalizer. Nothing persists inside a transport handler,
so the two paths cannot drift (which is how memory/traces/usage each got
"forgotten in the streaming path" before). See docs/aloy-send-message-refactor-spec.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from pori import (
    VIRTUAL_PREFIX,
    AgentMemory,
    ThreadData,
    get_thread_data,
    replace_virtual_path,
)

from .config import settings
from .memory_records import record_to_row, request_scope
from .models import (
    Conversation,
    CoreMemoryBlock,
    KnowledgeEntry,
    Message,
    Run,
    RunEventLog,
    StoredFile,
    TraceRecord,
    UsageRecord,
)
from .storage import artifact_key, get_object_store, safe_name
from .tenancy import OrganizationContext

logger = logging.getLogger("aloy_backend")


def _json_default(o: Any) -> Any:
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    return str(o)


def json_safe(value: Any) -> Any:
    """Normalize to plain JSON structures so JSON columns can store it.

    The agent's metrics/trace can carry rich objects (e.g. TokenUsage); the raw
    result object is authoritative but not always JSON-serializable, so coerce
    it once here rather than persisting objects that blow up on flush."""
    if value is None:
        return None
    return json.loads(json.dumps(value, default=_json_default))


# Backwards-compat alias (internal callers)
_json_safe = json_safe


def make_usage_record(
    *,
    organization_id: str,
    user_id: str,
    run_id: str,
    conversation_id: Optional[str],
    metrics: Optional[dict],
) -> Optional[UsageRecord]:
    """Build the UsageRecord for a run's metrics — the ONE mapping both the
    request paths and the durable worker use (field drift here is how billing
    silently broke before)."""
    if not metrics or not isinstance(metrics, dict):
        return None
    tokens = metrics.get("tokens") or {}
    return UsageRecord(
        organization_id=organization_id,
        user_id=user_id,
        run_id=run_id,
        conversation_id=conversation_id,
        provider=(metrics.get("model") or "").split("/")[0],
        model=(metrics.get("model") or "").split("/")[-1],
        input_tokens=int(tokens.get("input", 0)),
        output_tokens=int(tokens.get("output", 0)),
        total_tokens=int(tokens.get("total", 0)),
        estimated_cost=float((metrics.get("cost_usd") or "$0").replace("$", "") or 0),
    )


def make_trace_record(
    *,
    organization_id: str,
    user_id: str,
    run_id: str,
    conversation_id: Optional[str],
    trace: Optional[dict],
) -> Optional[TraceRecord]:
    """Build the TraceRecord for a run's trace — shared by all persist paths."""
    if not trace or not isinstance(trace, dict):
        return None
    return TraceRecord(
        organization_id=organization_id,
        user_id=user_id,
        run_id=run_id,
        conversation_id=conversation_id,
        trace_data=trace,
        duration_seconds=float((trace.get("duration") or "0s").replace("s", "") or 0),
        total_spans=int(trace.get("total_spans", 0)),
        status=trace.get("status", "ok"),
    )


async def flush_memory_to_db(
    session: AsyncSession, context: OrganizationContext, memory: AgentMemory
) -> None:
    """Stage core-memory + typed long-term memory changes (does NOT commit).

    Staged rows are committed by the finalizer's single commit — never commit
    here, or callers can't batch it into one transaction.
    """
    now = datetime.now(timezone.utc)
    for label in ("persona", "human", "notes"):
        block = memory.core_memory.get_block(label)
        existing = (
            (
                await session.execute(
                    select(CoreMemoryBlock).where(
                        CoreMemoryBlock.organization_id == context.organization_id,
                        CoreMemoryBlock.user_id == context.user_id,
                        CoreMemoryBlock.label == label,
                    )
                )
            )
            .scalars()
            .first()
        )
        if existing:
            if existing.value != block.value:
                existing.value = block.value
                existing.updated_at = now
                session.add(existing)
        elif block.value:
            session.add(
                CoreMemoryBlock(
                    organization_id=context.organization_id,
                    user_id=context.user_id,
                    label=label,
                    value=block.value,
                )
            )

    for record in memory.memory_records:
        if not request_scope(
            context.user_id,
            organization_id=context.organization_id,
            agent_id=memory.agent_id,
            session_id=memory.session_id,
        ).can_access(record.scope):
            continue
        existing_row = await session.get(KnowledgeEntry, record.id)
        session.add(record_to_row(record, existing_row))


@dataclass
class RunOutcome:
    """Everything needed to persist one finished agent turn, built once."""

    task: str
    final_answer: str
    reasoning: Optional[str]
    success: bool
    steps_taken: int
    metrics: Optional[dict]
    trace: Optional[dict]
    artifacts: list = field(default_factory=list)
    plan: list = field(default_factory=list)
    selected_skills: list = field(default_factory=list)
    # True when the user stopped the run mid-generation (final_answer is then
    # the partial streamed text, not a completed answer).
    stopped: bool = False
    # Identity (authoritative, from the run's RunContext).
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    organization_id: str = ""
    session_id: str = ""
    agent_id: str = "default_agent"
    # The run's memory, to flush; and the coalesced replay events (streaming).
    memory: Optional[AgentMemory] = None
    events: Optional[list] = None


def build_run_outcome(
    agent_result: dict,
    memory: Optional[AgentMemory],
    run_context: Any,
    task: str,
    *,
    fallback_org: str,
    events: Optional[list] = None,
) -> RunOutcome:
    """Map ``execute_task``'s return dict (same shape both paths get) into one
    ``RunOutcome``. The single place that knows the agent-result shape."""
    agent = agent_result.get("agent")
    final = agent.memory.get_final_answer() if agent else {}
    result_data = agent_result.get("result") or {}

    answer = (
        agent_result.get("final_answer")
        or (final or {}).get("final_answer")
        or "I could not generate a response."
    )
    return RunOutcome(
        task=task,
        final_answer=answer,
        stopped=bool(agent_result.get("stopped")),
        reasoning=agent_result.get("reasoning") or (final or {}).get("reasoning"),
        success=bool(agent_result.get("success")),
        steps_taken=int(agent_result.get("steps_taken") or 0),
        metrics=_json_safe(agent_result.get("metrics") or result_data.get("metrics")),
        trace=_json_safe(agent_result.get("trace") or result_data.get("trace")),
        artifacts=_json_safe(
            agent_result.get("artifacts") or result_data.get("artifacts") or []
        ),
        plan=_json_safe(agent_result.get("plan") or result_data.get("plan") or []),
        selected_skills=_json_safe(
            agent_result.get("selected_skills")
            or result_data.get("selected_skills")
            or []
        ),
        run_id=getattr(run_context, "run_id", None) or uuid.uuid4().hex,
        organization_id=getattr(run_context, "organization_id", None) or fallback_org,
        session_id=getattr(run_context, "session_id", None) or "",
        agent_id=getattr(run_context, "agent_id", None) or "default_agent",
        memory=(agent.memory if agent else None) or memory,
        events=events,
    )


def _resolve_artifact_file(path_str: str, thread: ThreadData) -> Optional[Path]:
    """Map an artifact's recorded path (virtual, relative, or absolute) to a
    real file inside the conversation's thread dirs. None = outside the jail
    (or a traversal attempt) — such entries are never uploaded."""
    raw = (path_str or "").strip()
    if not raw or raw == "(path unavailable)":
        return None
    try:
        if raw.startswith(VIRTUAL_PREFIX):
            return Path(replace_virtual_path(raw, thread))
        root = Path(thread.workspace_path).parent.resolve()  # .../user-data
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = Path(thread.workspace_path) / raw
        resolved = candidate.resolve()
        resolved.relative_to(root)  # containment or ValueError
        return resolved
    except ValueError:
        return None


def store_run_artifacts(
    session: AsyncSession,
    conv: Conversation,
    context: OrganizationContext,
    outcome: RunOutcome,
) -> None:
    """Extraction OUT: copy the run's written files from the (ephemeral)
    thread dirs into the object store, adding StoredFile pointer rows to the
    SAME session/transaction as the rest of the outcome — a disconnect can't
    orphan a blob. Entries gain file_id; oversize/missing files keep their
    pointer with a reason instead of silently vanishing."""
    artifacts = outcome.artifacts or []
    if not artifacts:
        return
    thread = get_thread_data(outcome.session_id or conv.id, settings.sandbox_base_dir)
    store = get_object_store()
    per_file_cap = settings.storage_max_artifact_mb * 1024 * 1024
    run_budget = settings.storage_max_run_artifact_mb * 1024 * 1024
    stored_by_path: dict[str, str] = {}  # resolved path -> file_id

    for entry in artifacts:
        if not isinstance(entry, dict) or entry.get("kind") != "file":
            continue
        real = _resolve_artifact_file(str(entry.get("path") or ""), thread)
        if real is None or not real.is_file():
            continue
        key_path = str(real)
        if key_path in stored_by_path:
            entry["file_id"] = stored_by_path[key_path]
            continue
        try:
            size = real.stat().st_size
            if size > per_file_cap:
                entry["too_large"] = True
                continue
            if size > run_budget:
                entry["over_run_budget"] = True
                continue
            digest = hashlib.sha256()
            with real.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    digest.update(chunk)
            file_id = uuid.uuid4().hex
            key = artifact_key(context.organization_id, conv.id, file_id, real.name)
            with real.open("rb") as fh:
                store.put(
                    key,
                    fh,
                    content_type=mimetypes.guess_type(real.name)[0]
                    or "application/octet-stream",
                )
            session.add(
                StoredFile(
                    id=file_id,
                    organization_id=context.organization_id,
                    user_id=context.user_id,
                    conversation_id=conv.id,
                    run_id=outcome.run_id,
                    kind="artifact",
                    name=safe_name(real.name),
                    content_type=mimetypes.guess_type(real.name)[0]
                    or "application/octet-stream",
                    size_bytes=size,
                    sha256=digest.hexdigest(),
                    storage_key=key,
                )
            )
            run_budget -= size
            stored_by_path[key_path] = file_id
            entry["file_id"] = file_id
        except Exception as exc:
            logger.exception(
                "Could not store artifact %r for run %s", key_path, outcome.run_id
            )
            # Persist the reason, not just a flag — a hidden/rotated server
            # log must never be the only witness to a storage failure.
            entry["storage_error"] = f"{type(exc).__name__}: {exc}"[:300]


async def persist_run_outcome(
    session: AsyncSession,
    conv: Conversation,
    context: OrganizationContext,
    outcome: RunOutcome,
) -> Message:
    """THE single finalizer: persist message + run + usage + trace + event-log +
    memory as one transaction. Both request paths call this; nothing else writes.
    Idempotent by run_id (a disconnect-finally + a retry can't double-write)."""
    existing_run = await session.get(Run, outcome.run_id)
    if existing_run is not None:
        # Already persisted (e.g. the disconnect-finally fired after a normal
        # finish). Return the existing assistant message.
        existing_msg = (
            (
                await session.execute(
                    select(Message)
                    .where(
                        Message.conversation_id == conv.id, Message.role == "assistant"
                    )
                    .order_by(col(Message.created_at).desc())
                )
            )
            .scalars()
            .first()
        )
        if existing_msg is not None:
            return existing_msg

    session_id = outcome.session_id or conv.id
    agent_id = outcome.agent_id or conv.agent_config_id or "default_agent"
    org_id = outcome.organization_id or context.organization_id
    trace = outcome.trace if isinstance(outcome.trace, dict) else None

    # Extraction OUT before the message row is built, so the artifact entries
    # persisted into metadata already carry their file_id pointers.
    store_run_artifacts(session, conv, context, outcome)

    assistant_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=outcome.final_answer,
        metadata_={
            "reasoning": outcome.reasoning,
            "steps_taken": outcome.steps_taken,
            "metrics": outcome.metrics,
            "selected_skills": outcome.selected_skills,
            "artifacts": outcome.artifacts,
            "plan": outcome.plan,
            "run_id": outcome.run_id,  # links to the replay log
            **({"stopped": True} if outcome.stopped else {}),
        },
    )
    session.add(assistant_msg)

    # This is a TERMINAL history record for a run that already executed inline
    # (streaming/blocking) — not a unit of work. The status MUST be terminal:
    # the durable worker's claim query treats status=='pending' (the Run model
    # default) as claimable, so an unset status here makes the worker re-execute
    # a finished run with max_steps=0 → ExecutionBudget(ge=1) crash. max_steps is
    # not applicable to an already-completed run; 0 is a deliberate sentinel that
    # is only safe because the status keeps this row out of the queue.
    run = Run(
        id=outcome.run_id,
        user_id=context.user_id,
        organization_id=org_id,
        agent_id=agent_id,
        session_id=session_id,
        conversation_id=conv.id,
        task=outcome.task,
        max_steps=0,
        status=(
            "cancelled"
            if outcome.stopped
            else "completed" if outcome.success else "failed"
        ),
        success=outcome.success,
        steps_taken=outcome.steps_taken,
        final_answer=outcome.final_answer,
        reasoning=outcome.reasoning,
        metrics=outcome.metrics,
        prompt_fingerprint=(trace or {}).get("prompt_fingerprint"),
        tool_surface_fingerprint=(trace or {}).get("tool_surface_fingerprint"),
        execution_receipts=(trace or {}).get("execution_receipts") or [],
        selected_skills=outcome.selected_skills,
        artifacts=outcome.artifacts,
        plan=outcome.plan,
    )
    session.add(run)

    usage = make_usage_record(
        organization_id=org_id,
        user_id=context.user_id,
        run_id=outcome.run_id,
        conversation_id=conv.id,
        metrics=outcome.metrics,
    )
    if usage is not None:
        session.add(usage)

    trace_record = make_trace_record(
        organization_id=org_id,
        user_id=context.user_id,
        run_id=outcome.run_id,
        conversation_id=conv.id,
        trace=trace,
    )
    if trace_record is not None:
        session.add(trace_record)

    if outcome.events:
        session.add(
            RunEventLog(
                run_id=outcome.run_id,
                organization_id=org_id,
                user_id=context.user_id,
                conversation_id=conv.id,
                events=outcome.events,
                event_count=len(outcome.events),
            )
        )

    if outcome.memory is not None:
        await flush_memory_to_db(session, context, outcome.memory)

    conv.updated_at = datetime.now(timezone.utc)
    session.add(conv)

    await session.commit()
    await session.refresh(assistant_msg)
    return assistant_msg
