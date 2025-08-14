import uuid
from fastapi import APIRouter, Depends

from pori.agent import AgentSettings
from pori.orchestrator import Orchestrator
from ..background import create_background_task, get_task_status
from ..deps import get_orchestrator
from ..models import (
    TaskCreateRequest,
    TaskCreateResponse,
    TaskStatusResponse,
    TaskResultResponse,
)

router = APIRouter()


@router.post("/tasks", response_model=TaskCreateResponse)
async def submit_task(
    request: TaskCreateRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> TaskCreateResponse:
    """
    Submit a new task for an agent to execute.
    """
    task_id = str(uuid.uuid4())

    agent_settings = AgentSettings(max_steps=request.max_steps)

    # Create a coroutine for the orchestrator to execute the task
    coro = orchestrator.execute_task(
        task=request.task,
        agent_settings=agent_settings,
    )

    # Run the coroutine in the background
    create_background_task(task_id, coro)

    return TaskCreateResponse(task_id=task_id, status="queued")


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status_endpoint(task_id: str):
    """
    Retrieves the status of a specific task.
    """
    status_info = get_task_status(task_id)
    return TaskStatusResponse(
        task_id=task_id,
        status=status_info.get("status"),
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
