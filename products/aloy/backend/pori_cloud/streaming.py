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
from typing import Any, AsyncGenerator, Optional, Set

from pori import AgentSettings, Orchestrator, RunContext
from pori.clarify import ClarificationRequest, ClarifyBridge
from pori.observability import RUN_END, PoriEvent

from .event_log import EventLogCollector

logger = logging.getLogger("pori_cloud")

# Active clarify bridges; the resolve endpoint routes an answer to whichever
# stream is awaiting it (see routes/conversations.py).
CLARIFY_BRIDGES: Set[ClarifyBridge] = set()

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
) -> AsyncGenerator[str, None]:
    queue: "asyncio.Queue[PoriEvent]" = asyncio.Queue()
    serving_loop = asyncio.get_running_loop()

    def push(event: PoriEvent) -> None:
        serving_loop.call_soon_threadsafe(queue.put_nowait, event)

    def emit_clarification(req: ClarificationRequest) -> None:
        push(PoriEvent("clarification_request", req.to_event(), step=0))

    bridge = ClarifyBridge(emit=emit_clarification)
    CLARIFY_BRIDGES.add(bridge)

    def run_agent() -> dict:
        return asyncio.run(
            orchestrator.execute_task(
                task=task,
                agent_settings=settings,
                run_context=run_context,
                on_event=push,
                stream=True,
                tool_context_extra={
                    "clarify_handler": bridge.as_sync_handler(serving_loop)
                },
            )
        )

    run = serving_loop.run_in_executor(None, run_agent)

    # `status` kept for pre-migration consumers; canonical stream is the PoriEvents.
    yield _sse_event("status", {"status": "running", "task": task})

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                if run.done() and queue.empty():
                    break
                yield ": keepalive\n\n"  # keep the socket open
                continue
            if isinstance(event, PoriEvent):
                # Record on the serving loop (single-threaded here) for the
                # read-only replay log — no race with the worker-thread push.
                if collector is not None:
                    try:
                        collector.record(event)
                    except Exception:  # logging must never break a run
                        logger.debug("Event log capture failed", exc_info=True)
                yield _porievent_frame(event)
                if event.type == RUN_END:
                    break

        try:
            result = await run
        except Exception as exc:  # run raised
            logger.exception("Streaming agent run failed")
            yield _sse_event("error", {"detail": str(exc)})
            return

        if not result:
            yield _sse_event("error", {"detail": "run produced no result"})
            return

        agent = result.get("agent")
        final = agent.memory.get_final_answer() if agent else None
        result_data = result.get("result", {}) or {}
        plan = result.get("plan") or result_data.get("plan") or []

        yield _sse_event(
            "message",
            {
                "role": "assistant",
                "content": (final or {}).get(
                    "final_answer", "I could not generate a response."
                ),
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

    except asyncio.CancelledError:
        yield _sse_event("error", {"detail": "Stream cancelled"})
        raise
    except Exception as exc:
        logger.exception("Streaming error")
        yield _sse_event("error", {"detail": str(exc)})
    finally:
        bridge.cancel_pending()  # unblock a waiting ask_user on disconnect
        CLARIFY_BRIDGES.discard(bridge)
        yield _sse_event("done", {})


def resolve_clarification(clarification_id: str, value: str) -> bool:
    """Deliver a clarify answer to whichever active stream awaits it."""
    for bridge in list(CLARIFY_BRIDGES):
        if bridge.submit_answer(clarification_id, value):
            return True
    return False
