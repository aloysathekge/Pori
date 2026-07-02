from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TaskCreateRequest(BaseModel):
    task: str
    max_steps: int = 50
    stream: bool = False  # Reserved for future SSE/WebSocket implementation


class TaskCreateResponse(BaseModel):
    task_id: str
    status: Literal["queued", "running", "completed", "failed"]
    submitted_at: datetime = Field(default_factory=datetime.now)


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    details: str | None = None


class TaskResultResponse(BaseModel):
    task_id: str
    success: bool
    final_answer: str | None = None
    reasoning: str | None = None
