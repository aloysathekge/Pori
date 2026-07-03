import asyncio
import json
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from pori.agent import AgentSettings
from pori.memory import AgentMemory
from pori.observability import RUN_END, PoriEvent
from pori.orchestrator import Orchestrator

from ..background import create_background_task, get_task_status
from ..deps import get_orchestrator, get_request_memory
from ..models import (
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
) -> StreamingResponse:
    """Run a task and stream normalized PoriEvents as Server-Sent Events (GW-4).

    Turns the poll-only API real-time over the same event contract the CLI already
    consumes. The stream ends on RUN_END (or when the run otherwise finishes); a
    client disconnect cancels the run.
    """
    queue: "asyncio.Queue[PoriEvent]" = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_event(event: PoriEvent) -> None:
        # call_soon_threadsafe keeps this correct even if the run later moves to a
        # worker thread (GW-6); it's harmless when the run is already on the loop.
        loop.call_soon_threadsafe(queue.put_nowait, event)

    run = asyncio.create_task(
        orchestrator.execute_task(
            task=request.task,
            agent_settings=AgentSettings(max_steps=request.max_steps),
            memory=memory,
            on_event=on_event,
            stream=True,
        )
    )

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
            if not run.done():
                run.cancel()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
