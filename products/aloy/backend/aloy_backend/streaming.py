"""SSE streaming for agent execution — kernel ``PoriEvent`` contract + clarify.

Relays the kernel's live ``PoriEvent`` stream as SSE (``run_start`` / ``step_*`` /
``text_delta`` / ``thinking_delta`` / ``tool_call_start|end`` / ``run_end``), plus
a final ``message`` frame with the answer + metrics for DB persistence.

The run executes on its own loop **in a worker thread**, so a blocking
``ask_user`` (waiting on a clarify button over HTTP) never blocks the serving
loop. A :class:`ClarifyBridge` emits ``clarification_request`` frames and pauses
the run until the client POSTs the answer, which the resolve endpoint routes via
the module-level :data:`CLARIFY_BRIDGES` registry.

Harvested from the kernel ``pori/api`` stream_task (worker-thread + bridge model).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional

from pori import (
    RUN_END,
    STEP_START,
    TEXT_DELTA,
    AgentSettings,
    CancellationToken,
    ClarificationRequest,
    ClarifyBridge,
    Orchestrator,
    PoriEvent,
    RunContext,
)

from . import live_runs, resumable_runs
from .approvals import (
    APPROVAL_BRIDGES,
    ApprovalBridge,
    build_write_hitl_config,
    proposal_write_gate,
)
from .event_log import EventLogCollector
from .tools import GOOGLE_WRITE_TOOLS, gmail_draft_preview

logger = logging.getLogger("aloy_backend")

# Active clarify bridges, keyed to the (organization_id, user_id) that owns the
# run — the resolve endpoint only routes an answer to the caller's OWN bridges,
# so one user can never answer another user's clarification.
# NOTE: in-process registry — a resolve must land on the worker holding the
# bridge (single-worker constraint; move to a shared store to scale out).
CLARIFY_BRIDGES: Dict[ClarifyBridge, tuple] = {}

_KEEPALIVE_SECONDS = 15


def _safe_json(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=_safe_json)}\n\n"


def _porievent_frame(event: PoriEvent) -> str:
    """A kernel ``PoriEvent`` as an SSE frame (matches ``@aloy/shared``)."""
    return _sse_event(event.type, {"payload": event.payload, "step": event.step})


async def stream_agent_execution(
    orchestrator: Orchestrator,
    task: str,
    settings: AgentSettings,
    run_context: Optional[RunContext] = None,
    collector: Optional[EventLogCollector] = None,
    tool_context_extra: Optional[dict] = None,
    mcp_servers: Optional[list] = None,
    task_attachments: Optional[list] = None,
    result_holder: Optional[dict] = None,
    conversation_id: Optional[str] = None,
    resume_task_id: Optional[str] = None,
    sandbox_base_dir: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Run the agent and stream its frames.

    Pump architecture: a background PUMP task owns the run's whole event
    lifecycle (drain events, record for replay, publish frames to the live-run
    hub, build the final message frame, clean up the clarify bridge). The HTTP
    response below is just the FIRST SUBSCRIBER — if the client disconnects,
    the pump keeps going, and a returning client re-attaches through the live
    endpoint with full replay.
    """
    queue: "asyncio.Queue[PoriEvent]" = asyncio.Queue()
    serving_loop = asyncio.get_running_loop()

    def push(event: PoriEvent) -> None:
        serving_loop.call_soon_threadsafe(queue.put_nowait, event)

    def emit_clarification(req: ClarificationRequest) -> None:
        push(PoriEvent("clarification_request", req.to_event(), step=0))

    owner = (
        getattr(run_context, "organization_id", None) or "",
        getattr(run_context, "user_id", None) or "",
    )
    bridge = ClarifyBridge(emit=emit_clarification)
    CLARIFY_BRIDGES[bridge] = owner

    merged_tool_context = dict(tool_context_extra or {})
    merged_tool_context["clarify_handler"] = bridge.as_sync_handler(serving_loop)

    def emit_approval(event: dict) -> None:
        push(PoriEvent("approval_request", event, step=0))

    def enrich_approval(tool: str, arguments: dict) -> dict:
        # A "send this draft" gate carries only a draft id — fetch the draft's
        # to/subject/body so the user reviews the real email, not an opaque id.
        if tool == "gmail_send_draft" and arguments.get("draft_id"):
            return gmail_draft_preview(merged_tool_context, str(arguments["draft_id"]))
        return {}

    approval_bridge: ApprovalBridge | None = None
    if run_context is not None and run_context.event_id:
        hitl_handler, hitl_config = proposal_write_gate(
            run_context=run_context,
            tools_registry=orchestrator.tools_registry,
            emit=emit_approval,
            enrich=enrich_approval,
            owner_loop=serving_loop,
        )
    else:
        # Kernel-only and legacy callers have no Event aggregate to own a Proposal.
        approval_bridge = ApprovalBridge(emit=emit_approval, enrich=enrich_approval)
        APPROVAL_BRIDGES[approval_bridge] = owner
        hitl_handler = approval_bridge
        hitl_config = build_write_hitl_config(GOOGLE_WRITE_TOOLS)

    cancel_token = CancellationToken()

    def run_agent() -> dict:
        return asyncio.run(
            orchestrator.execute_task(
                task=task,
                agent_settings=settings,
                run_context=run_context,
                on_event=push,
                stream=True,
                tool_context_extra=merged_tool_context,
                mcp_servers=mcp_servers,
                task_attachments=task_attachments,
                cancellation_token=cancel_token,
                resume_task_id=resume_task_id,
                sandbox_base_dir=sandbox_base_dir,
                hitl_handler=hitl_handler,
                hitl_config=hitl_config,
            )
        )

    run = serving_loop.run_in_executor(None, run_agent)
    # Expose the in-flight future: if the client disconnects mid-run the
    # response generator dies, but the pump + agent keep working — the caller
    # awaits this in the background and persists the outcome.
    if result_holder is not None:
        result_holder["run_future"] = run

    def request_stop() -> None:
        # Cooperative stop: the agent loop halts at the next step boundary. A
        # run blocked on a pending clarification never reaches that boundary,
        # so unblock it too (empty answer → the loop resumes → sees the token).
        cancel_token.cancel()
        bridge.cancel_pending()
        if approval_bridge is not None:
            approval_bridge.cancel_pending()

    run_id = getattr(run_context, "run_id", "") or ""
    live = live_runs.register(conversation_id or run_id, run_id, cancel=request_stop)

    async def pump() -> None:
        try:
            live.publish(_sse_event("status", {"status": "running", "task": task}))
            await _pump_events()
        except Exception as exc:
            # The pump must never die silently — every consumer would just see
            # the stream end with no message.
            logger.exception("Stream pump failed")
            live.publish(_sse_event("error", {"detail": str(exc)}))
        finally:
            # The RUN owns the clarify bridge lifecycle (not the HTTP response):
            # a re-attached client can still answer a pending clarification.
            bridge.cancel_pending()
            if approval_bridge is not None:
                approval_bridge.cancel_pending()
            CLARIFY_BRIDGES.pop(bridge, None)
            if approval_bridge is not None:
                APPROVAL_BRIDGES.pop(approval_bridge, None)
            live.publish(_sse_event("done", {}))
            live.finish()
            live_runs.retire_later(conversation_id or run_id, live)

    async def _pump_events() -> None:
        # The current step's streamed prose. If the user stops the run
        # mid-generation, this is what they watched being written — keep it as
        # the interrupted answer instead of throwing it away (the aborted LLM
        # call never returns, so the kernel can't hand it back).
        partial_text: list = []
        while True:
            # Runs that end without a RUN_END event (crash, legacy path) would
            # otherwise idle a full keepalive interval before we notice.
            if run.done() and queue.empty():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                continue
            if isinstance(event, PoriEvent):
                if event.type == STEP_START:
                    partial_text.clear()
                elif event.type == TEXT_DELTA:
                    partial_text.append((event.payload or {}).get("text") or "")
                # Record on the serving loop (single-threaded here) for the
                # read-only replay log — no race with the worker push.
                if collector is not None:
                    try:
                        collector.record(event)
                    except Exception:
                        logger.debug("Event log capture failed", exc_info=True)
                live.publish(_porievent_frame(event))
                if event.type == RUN_END:
                    break

        try:
            result = await run
        except Exception as exc:
            logger.exception("Streaming agent run failed")
            live.publish(_sse_event("error", {"detail": str(exc)}))
            return
        if not result:
            live.publish(_sse_event("error", {"detail": "run produced no result"}))
            return
        # The authoritative result object, for the caller's finalizer.
        if result_holder is not None:
            result_holder["result"] = result

        agent = result.get("agent")
        final = agent.memory.get_final_answer() if agent else None
        # A stopped run has no final answer — keep the partial text the user
        # watched being generated (Claude-style interrupted response), writing
        # it into the authoritative result so persistence shows the same thing.
        if cancel_token.cancelled and not (
            result.get("final_answer") or (final or {}).get("final_answer")
        ):
            result["final_answer"] = (
                "".join(partial_text).strip() or "*(stopped before any output)*"
            )
            result["stopped"] = True
            # Park the run's warm state so a "continue" can truly resume the
            # kernel task from its checkpoint instead of starting over.
            if agent is not None and conversation_id:
                resumable_runs.register(
                    conversation_id,
                    resumable_runs.ResumableRun(
                        run_id=getattr(run_context, "run_id", "") or "",
                        task=task,
                        task_id=agent.task_id,
                        memory=agent.memory,
                    ),
                )
        result_data = result.get("result", {}) or {}
        plan = result.get("plan") or result_data.get("plan") or []
        live.publish(
            _sse_event(
                "message",
                {
                    "role": "assistant",
                    "run_id": getattr(run_context, "run_id", None),
                    "content": (
                        result.get("final_answer")
                        or (final or {}).get("final_answer")
                        or "I could not generate a response."
                    ),
                    "stopped": bool(result.get("stopped")) or None,
                    "reasoning": (final or {}).get("reasoning"),
                    "steps_taken": int(result.get("steps_taken") or 0),
                    "success": bool(result.get("success")),
                    "metrics": result_data.get("metrics"),
                    "run_context": result_data.get("run_context"),
                    "trace": result.get("trace"),
                    "selected_skills": (
                        result.get("selected_skills")
                        or result_data.get("selected_skills")
                        or []
                    ),
                    "artifacts": (
                        result.get("artifacts") or result_data.get("artifacts") or []
                    ),
                    "plan": plan,
                },
            )
        )

    pump_task = serving_loop.create_task(pump())
    if result_holder is not None:
        result_holder["pump_task"] = pump_task

    # ---- first subscriber: this HTTP response ----
    async for frame in subscribe_frames(live):
        yield frame


async def subscribe_frames(live: "live_runs.LiveRun") -> AsyncGenerator[str, None]:
    """Replay a live run's buffered frames, then follow it until done."""
    replay, q = live.subscribe()
    try:
        for frame in replay:
            yield frame
        while True:
            try:
                frame = await asyncio.wait_for(q.get(), timeout=_KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                # If the run just finished, our queue already holds the
                # remaining frames + the None sentinel (finish() guarantees
                # it) — loop back and drain instead of breaking early.
                if not live.done:
                    yield ": keepalive\n\n"
                continue
            if frame is None:  # end-of-stream sentinel
                break
            yield frame
    finally:
        live.unsubscribe(q)


def resolve_clarification(
    clarification_id: str,
    value: str,
    *,
    organization_id: str,
    user_id: str,
) -> bool:
    """Deliver a clarify answer to the CALLER'S awaiting stream.

    Only bridges owned by (organization_id, user_id) are considered, so a valid
    clarification id belonging to another user resolves to False (404 at the
    endpoint) instead of injecting text into their live run."""
    owner = (organization_id, user_id)
    for bridge, bridge_owner in list(CLARIFY_BRIDGES.items()):
        if bridge_owner != owner:
            continue
        if bridge.submit_answer(clarification_id, value):
            return True
    return False
