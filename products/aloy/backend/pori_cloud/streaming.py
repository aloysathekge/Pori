"""SSE streaming for agent execution — the kernel ``PoriEvent`` contract.

Relays the kernel's live ``PoriEvent`` stream (``run_start`` / ``step_*`` /
``text_delta`` / ``thinking_delta`` / ``tool_call_start`` / ``tool_call_end`` /
``run_end`` …) as SSE frames — the same contract the kernel API and
``@aloy/shared`` speak. Replaces the earlier step-polling approach, so the web
surface gets real token streaming + tool events (and, once the clarify bridge is
wired, ``clarification_request`` buttons).

A final ``message`` event is still emitted after the run with the authoritative
answer + metrics, so ``conversations.py`` persistence (which captures
``event: message``) is unchanged, and a non-streaming model still yields a final
answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Optional

from pori import AgentSettings, Orchestrator, RunContext
from pori.observability import PoriEvent

logger = logging.getLogger("pori_cloud")

_SENTINEL = object()


def _safe_json(obj: Any) -> Any:
    """JSON serializer that handles pydantic models and dataclasses."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, default=_safe_json)}\n\n"


def _porievent_frame(event: PoriEvent) -> str:
    """A kernel ``PoriEvent`` as an SSE frame (matches ``@aloy/shared``)."""
    return _sse_event(event.type, {"payload": event.payload, "step": event.step})


async def stream_agent_execution(
    orchestrator: Orchestrator,
    task: str,
    settings: AgentSettings,
    run_context: Optional[RunContext] = None,
) -> AsyncGenerator[str, None]:
    """Run the agent and yield its ``PoriEvent`` stream as SSE, then a final
    ``message`` event with the answer + metrics for persistence."""
    queue: asyncio.Queue = asyncio.Queue()
    holder: dict = {}

    def on_event(event: PoriEvent) -> None:
        # execute_task runs on this event loop, so put_nowait is safe.
        queue.put_nowait(event)

    async def _run() -> None:
        try:
            holder["result"] = await orchestrator.execute_task(
                task=task,
                agent_settings=settings,
                run_context=run_context,
                on_event=on_event,
            )
        except Exception as exc:  # captured, surfaced as an error frame
            holder["error"] = exc
            logger.exception("Streaming agent run failed")
        finally:
            queue.put_nowait(_SENTINEL)

    agent_task = asyncio.create_task(_run())

    # `status` kept for backward compatibility with pre-migration consumers;
    # the canonical stream is the relayed PoriEvents below.
    yield _sse_event("status", {"status": "running", "task": task})

    try:
        while True:
            event = await queue.get()
            if event is _SENTINEL:
                break
            if isinstance(event, PoriEvent):
                yield _porievent_frame(event)

        result = holder.get("result")
        if result is None:
            yield _sse_event(
                "error", {"detail": str(holder.get("error") or "run failed")}
            )
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
        agent_task.cancel()
        yield _sse_event("error", {"detail": "Stream cancelled"})
        raise
    except Exception as exc:
        logger.exception("Streaming error")
        yield _sse_event("error", {"detail": str(exc)})
    finally:
        yield _sse_event("done", {})
