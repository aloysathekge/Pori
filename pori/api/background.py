import asyncio
from typing import Any, Coroutine, Dict

# A simple in-memory store for running tasks.
# In a production system, this would be replaced with Redis, a database, or another persistent store.
_background_tasks: Dict[str, asyncio.Task] = {}


def create_background_task(task_id: str, coro: Coroutine) -> asyncio.Task:
    """
    Creates and stores a background task.
    """
    task = asyncio.create_task(coro)
    _background_tasks[task_id] = task
    return task


def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Retrieves the status of a background task.
    """
    task = _background_tasks.get(task_id)
    if not task:
        return {"status": "not_found"}

    if task.done():
        try:
            result = task.result()
            return {"status": "completed", "result": result}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    else:
        return {"status": "running"}
