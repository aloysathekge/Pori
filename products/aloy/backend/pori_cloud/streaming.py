"""SSE streaming for agent execution."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator

from pori import AgentMemory, AgentSettings, Orchestrator, RunContext
from pori.observability import build_tool_preview

logger = logging.getLogger("pori_cloud")


def _safe_json(obj):
    """JSON serializer that handles pydantic models and dataclasses."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, default=_safe_json)}\n\n"


async def stream_agent_execution(
    orchestrator: Orchestrator,
    task: str,
    settings: AgentSettings,
    poll_interval: float = 0.3,
    run_context: RunContext | None = None,
) -> AsyncGenerator[str, None]:
    """
    Run the agent and yield SSE events as it progresses.

    Yields step_start/step_end events by polling agent state,
    then a final message/error event when done.
    """
    # Track state for change detection
    last_step = 0
    agent_ref: dict = {}

    def _agent_plan(agent) -> list[dict]:
        """Serialize the model-owned plan (update_plan) for streaming."""
        store = getattr(agent, "plan_store", None)
        return [item.model_dump() for item in store.items()] if store else []

    async def _run():
        result = await orchestrator.execute_task(
            task=task,
            agent_settings=settings,
            run_context=run_context,
        )
        agent_ref["result"] = result

    # Start agent in background
    agent_task = asyncio.create_task(_run())

    yield _sse_event("status", {"status": "running", "task": task})

    try:
        while not agent_task.done():
            # Find the active agent (most recently added)
            if orchestrator.agents:
                task_id = list(orchestrator.agents.keys())[-1]
                agent = orchestrator.agents[task_id]
                current_step = agent.state.n_steps

                if current_step > last_step:
                    # Emit events for steps we missed
                    for step in range(last_step + 1, current_step + 1):
                        # Get the latest tool call info if available
                        tool_info = None
                        if agent.memory.tool_call_history:
                            recent = agent.memory.tool_call_history[-1]
                            tool_info = {
                                "tool": recent.tool_name,
                                "success": recent.success,
                                # Mechanical detail line, e.g. "Running: npm run build"
                                "preview": build_tool_preview(
                                    recent.tool_name, recent.parameters
                                ),
                            }

                        yield _sse_event(
                            "step",
                            {
                                "step": step,
                                "max_steps": settings.max_steps,
                                # Model-authored intent line (the agent's next_goal)
                                "activity": getattr(
                                    agent.state, "current_activity", ""
                                ),
                                "tool": tool_info,
                                "plan": _agent_plan(agent),
                            },
                        )
                    last_step = current_step

            await asyncio.sleep(poll_interval)

        # Agent finished — get result
        result = agent_ref.get("result")
        if result is None:
            # Task raised an exception
            exc = agent_task.exception()
            yield _sse_event("error", {"detail": str(exc)})
            return

        agent = result.get("agent")
        final = agent.memory.get_final_answer() if agent else None
        metrics = result.get("result", {}).get("metrics")
        result_data = result.get("result", {})
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
                "metrics": metrics,
                "run_context": result_data.get("run_context"),
                "trace": result.get("trace"),
                "selected_skills": result.get("selected_skills")
                or result_data.get("selected_skills")
                or [],
                "artifacts": result.get("artifacts")
                or result_data.get("artifacts")
                or [],
                "plan": plan,
            },
        )

    except asyncio.CancelledError:
        agent_task.cancel()
        yield _sse_event("error", {"detail": "Stream cancelled"})
    except Exception as e:
        logger.exception("Streaming error")
        yield _sse_event("error", {"detail": str(e)})
    finally:
        yield _sse_event("done", {})
