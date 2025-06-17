from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class AgentMessage(BaseModel):
    """A message in the agent's conversation history."""

    role: str  # 'system', 'user', 'assistant', etc.
    content: str
    timestamp: datetime = None

    def __init__(self, **data):
        if "timestamp" not in data:
            data["timestamp"] = datetime.now()
        super().__init__(**data)


class ToolCall(BaseModel):
    """Record of a tool call made by the agent."""

    tool_name: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    timestamp: datetime = None

    def __init__(self, **data):
        if "timestamp" not in data:
            data["timestamp"] = datetime.now()
        super().__init__(**data)


class TaskState(BaseModel):
    """State of a task being executed by the agent."""

    task_id: str
    description: str
    status: str = "in_progress"  # 'in_progress', 'completed', 'failed'
    created_at: datetime = None
    completed_at: Optional[datetime] = None

    def __init__(self, **data):
        if "created_at" not in data:
            data["created_at"] = datetime.now()
        super().__init__(**data)

    def complete(self, success: bool = True):
        """Mark the task as completed."""
        self.status = "completed" if success else "failed"
        self.completed_at = datetime.now()


class AgentMemory:
    """Memory system for the agent."""

    def __init__(self):
        self.conversation_history: List[AgentMessage] = []
        self.tool_call_history: List[ToolCall] = []
        self.tasks: Dict[str, TaskState] = {}
        self.state: Dict[str, Any] = {}
        self.summaries: List[Dict[str, Any]] = []
        # Add a specific property to track final answer
        self.final_answer: Optional[Dict[str, str]] = None

    # Add a method to get the final answer
    def get_final_answer(self) -> Optional[Dict[str, str]]:
        """Get the final answer if one exists."""
        return self.get_state("final_answer")

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        message = AgentMessage(role=role, content=content)
        self.conversation_history.append(message)

    def add_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Dict[str, Any],
        success: bool,
    ) -> None:
        """Record a tool call."""
        tool_call = ToolCall(
            tool_name=tool_name, parameters=parameters, result=result, success=success
        )
        self.tool_call_history.append(tool_call)

    def create_task(self, task_id: str, description: str) -> TaskState:
        """Create a new task."""
        task = TaskState(task_id=task_id, description=description)
        self.tasks[task_id] = task
        return task

    def update_state(self, key: str, value: Any) -> None:
        """Update the agent's state."""
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a value from the agent's state."""
        return self.state.get(key, default)

    def create_summary(self, step: int, max_messages: int = 10) -> str:
        """Create a summary of recent activity to prevent context overflow."""
        recent_messages = self.conversation_history[-max_messages:]
        recent_tools = self.tool_call_history[-max_messages:]

        # In a real implementation, you might use the LLM to generate this summary
        summary = {
            "step": step,
            "timestamp": datetime.now(),
            "message_count": len(self.conversation_history),
            "tool_call_count": len(self.tool_call_history),
            "current_tasks": {id: task.status for id, task in self.tasks.items()},
        }

        self.summaries.append(summary)
        return (
            f"Step {step} summary: {len(self.conversation_history)} messages, "
            f"{len(self.tool_call_history)} tool calls, "
            f"{len([t for t in self.tasks.values() if t.status == 'in_progress'])} tasks in progress"
        )
