import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from pori.agent import AgentSettings
from pori.clarify import ClarificationRequest, ClarifyBridge
from pori.memory import AgentMemory
from pori.observability import RUN_END, PoriEvent
from pori.orchestrator import Orchestrator

from ..background import create_background_task, get_task_status
from ..deps import get_clarify_bridges, get_orchestrator, get_request_memory
from ..models import (
    ClarifyAnswer,
    TaskCreateRequest,
    TaskCreateResponse,
    TaskResultResponse,
    TaskStatusResponse,
)
from ..security import get_api_key

router = APIRouter(dependencies=[Depends(get_api_key)])


@router.post("/tasks", response_model=TaskCreateResponse)
async def submit_task(
    request: TaskCreateRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    memory: AgentMemory = Depends(get_request_memory),
) -> TaskCreateResponse:
    """
    Submit a new task for an agent to execute.
    """
    task_id = str(uuid.uuid4())

    agent_settings = AgentSettings(max_steps=request.max_steps)

    # Create a coroutine for the orchestrator to execute the task.
    # Pass the per-request memory so concurrent callers stay isolated
    # (each request has its own session/namespace; only the store is shared).
    coro = orchestrator.execute_task(
        task=request.task,
        agent_settings=agent_settings,
        memory=memory,
    )

    # Run the coroutine in the background
    create_background_task(task_id, coro)

    return TaskCreateResponse(task_id=task_id, status="queued")


_KEEPALIVE_SECONDS = 15


def _sse_frame(event: PoriEvent) -> str:
    data = json.dumps({"payload": event.payload, "step": event.step})
    return f"event: {event.type}\ndata: {data}\n\n"


@router.post("/tasks/stream")
async def stream_task(
    request: TaskCreateRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    memory: AgentMemory = Depends(get_request_memory),
    bridges: set = Depends(get_clarify_bridges),
) -> StreamingResponse:
    """Run a task and stream normalized PoriEvents as Server-Sent Events (GW-4).

    Turns the poll-only API real-time over the same event contract the CLI already
    consumes. When the agent calls ask_user with options, a ``clarification_request``
    frame is emitted (the client renders buttons) and the run pauses until the
    client POSTs the answer to ``/v1/clarify/{id}``. The stream ends on RUN_END.

    The run executes on its own loop in a worker thread, so a blocking ask_user
    (waiting on a button-tap over HTTP) never blocks the serving loop.
    """
    queue: "asyncio.Queue[PoriEvent]" = asyncio.Queue()
    serving_loop = asyncio.get_running_loop()

    def push(event: PoriEvent) -> None:
        serving_loop.call_soon_threadsafe(queue.put_nowait, event)

    def emit_clarification(req: ClarificationRequest) -> None:
        push(PoriEvent("clarification_request", req.to_event(), step=0))

    bridge = ClarifyBridge(emit=emit_clarification)
    bridges.add(bridge)

    def run_agent():
        return asyncio.run(
            orchestrator.execute_task(
                task=request.task,
                agent_settings=AgentSettings(max_steps=request.max_steps),
                memory=memory,
                on_event=push,
                stream=True,
                tool_context_extra={
                    "clarify_handler": bridge.as_sync_handler(serving_loop)
                },
            )
        )

    run = serving_loop.run_in_executor(None, run_agent)

    async def event_stream():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=_KEEPALIVE_SECONDS
                    )
                except asyncio.TimeoutError:
                    if run.done() and queue.empty():
                        break
                    yield ": keepalive\n\n"  # comment frame keeps the socket open
                    continue
                yield _sse_frame(event)
                if event.type == RUN_END:
                    break
        finally:
            bridge.cancel_pending()  # unblock a waiting ask_user on disconnect
            bridges.discard(bridge)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/clarify/{clarification_id}")
async def submit_clarification(
    clarification_id: str,
    answer: ClarifyAnswer,
    bridges: set = Depends(get_clarify_bridges),
) -> dict:
    """Resume a paused run by delivering the user's clarification answer (a tapped
    option or free text). Routes to whichever active stream is awaiting it."""
    for bridge in list(bridges):
        if bridge.submit_answer(clarification_id, answer.value):
            return {"ok": True}
    raise HTTPException(
        status_code=404, detail="Unknown or already-answered clarification"
    )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status_endpoint(task_id: str):
    """
    Retrieves the status of a specific task.
    """
    status_info = get_task_status(task_id)
    return TaskStatusResponse(
        task_id=task_id,
        status=status_info.get("status") or "unknown",
        details=status_info.get("error"),
    )


@router.get("/tasks/{task_id}/result", response_model=TaskResultResponse)
async def get_task_result_endpoint(
    task_id: str, orchestrator: Orchestrator = Depends(get_orchestrator)
):
    """
    Retrieves the final result of a completed task.
    """
    status_info = get_task_status(task_id)

    if status_info.get("status") != "completed":
        return TaskResultResponse(
            task_id=task_id, success=False, final_answer="Task is not complete."
        )

    # The result from the background task is the agent's final return value
    agent_result = status_info.get("result", {})
    agent = agent_result.get("agent")

    if not agent:
        return TaskResultResponse(
            task_id=task_id, success=False, final_answer="Agent not found in result."
        )

    final_answer = agent.memory.get_final_answer()

    if not final_answer:
        return TaskResultResponse(
            task_id=task_id,
            success=False,
            final_answer="Final answer not found in memory.",
        )

    return TaskResultResponse(
        task_id=task_id,
        success=True,
        final_answer=final_answer.get("final_answer"),
        reasoning=final_answer.get("reasoning"),
    )
