from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TaskCreateRequest(BaseModel):
    task: str
    max_steps: int = 50
    stream: bool = False  # legacy flag; for streaming, POST /v1/tasks/stream (SSE)


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


class ClarifyAnswer(BaseModel):
    """The user's answer to a clarification (a tapped option or free text)."""

    value: str
