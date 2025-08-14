import uuid
from fastapi import APIRouter, Depends

from pori.agent import AgentSettings
from pori.orchestrator import Orchestrator
from ..background import create_background_task
from ..deps import get_orchestrator
from ..models import TaskCreateRequest, TaskCreateResponse

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
