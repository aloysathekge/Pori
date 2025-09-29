"""
Memory system for Pori agents.

Provides conversation history, tool call tracking, task management,
long-term experience storage, and semantic recall capabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
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


class ToolCallRecord(BaseModel):
    """Record of a tool call made by the agent."""
    
    tool_name: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    timestamp: datetime = None
    task_id: Optional[str] = None
    
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


class SimpleMemory:
    """
    Primary memory system for Pori agents.
    
    Provides conversation history, tool execution tracking, task state management,
    experience storage, and basic semantic recall. Designed for reliability
    and ease of extension.
    """
    
    def __init__(self):
        """Initialize a new session memory."""
        # Core memory components
        self.messages: List[AgentMessage] = []
        self.tool_call_history: List[ToolCallRecord] = []
        self.tasks: Dict[str, TaskState] = {}
        self.state: Dict[str, Any] = {}
        self.summaries: List[Dict[str, Any]] = []
        
        # Track current task context
        self.current_task_id: Optional[str] = None
        
        # Simple experience storage for this session
        self.experiences: List[Dict[str, Any]] = []
        
    # ============= Core Memory Methods =============
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        message = AgentMessage(role=role, content=content)
        self.messages.append(message)
    
    def get_recent_messages(self, n: int = 10) -> str:
        """Get the last n messages as a string."""
        recent = self.messages[-n:]
        return "\n".join([f"{msg.role}: {msg.content}" for msg in recent])
    
    def get_recent_messages_structured(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the last n messages as structured dictionaries."""
        recent = self.messages[-n:]
        return [
            {"role": msg.role, "content": msg.content}
            for msg in recent
        ]
    
    # ============= Tool Call Tracking =============
    
    def add_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Dict[str, Any],
        success: bool,
    ) -> None:
        """Record a tool call."""
        tool_call = ToolCallRecord(
            tool_name=tool_name,
            parameters=parameters,
            result=result,
            success=success,
            task_id=self.current_task_id,
        )
        self.tool_call_history.append(tool_call)
    
    # ============= Task Management =============
    
    def create_task(self, task_id: str, description: str) -> TaskState:
        """Create a new task and set it as current."""
        task = TaskState(task_id=task_id, description=description)
        self.tasks[task_id] = task
        self.begin_task(task_id)
        return task
    
    def begin_task(self, task_id: str) -> None:
        """Set the current task context."""
        self.current_task_id = task_id
        # Clear task-specific state
        self.state.pop("final_answer", None)
    
    # ============= State Management =============
    
    def update_state(self, key: str, value: Any) -> None:
        """Update the agent's state."""
        self.state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a value from the agent's state."""
        return self.state.get(key, default)
    
    def get_final_answer(self) -> Optional[Dict[str, str]]:
        """Get the final answer if one exists."""
        return self.get_state("final_answer")
    
    # ============= Summarization =============
    
    def create_summary(self, step: int, max_messages: int = 10) -> str:
        """Create a summary of recent activity."""
        summary = {
            "step": step,
            "timestamp": datetime.now(),
            "message_count": len(self.messages),
            "tool_call_count": len(self.tool_call_history),
            "current_tasks": {id: task.status for id, task in self.tasks.items()},
        }
        
        self.summaries.append(summary)
        
        # Create a text summary for the agent
        active_tasks = [t for t in self.tasks.values() if t.status == "in_progress"]
        return (
            f"Step {step} summary: {len(self.messages)} messages, "
            f"{len(self.tool_call_history)} tool calls, "
            f"{len(active_tasks)} tasks in progress"
        )
    
    # ============= Simple Experience Storage =============
    
    def add_experience(
        self, 
        text: str, 
        importance: int = 1, 
        meta: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store an experience for this session."""
        exp_id = f"exp_{len(self.experiences)}"
        experience = {
            "id": exp_id,
            "text": text,
            "importance": importance,
            "meta": meta or {},
            "timestamp": datetime.now(),
        }
        self.experiences.append(experience)
        return exp_id
    
    def recall(
        self, 
        query: str, 
        k: int = 5, 
        min_score: float = 0.0
    ) -> List[Tuple[str, str, float]]:
        """
        Simple recall based on keyword matching.
        Returns list of (id, text, score) tuples.
        """
        # For simplicity, return recent experiences
        # In a real implementation, you could add simple keyword matching
        results = []
        query_lower = query.lower()
        
        for exp in self.experiences[-k*2:]:  # Look at recent experiences
            text = exp["text"]
            # Simple keyword matching score
            score = 0.0
            if query_lower in text.lower():
                score = 1.0
            elif any(word in text.lower() for word in query_lower.split()):
                score = 0.5
            
            if score >= min_score:
                results.append((exp["id"], text, score))
        
        # Sort by score and return top k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:k]
    
    # ============= Backwards Compatibility =============
    
    @property
    def conversation_history(self) -> List[AgentMessage]:
        """Backwards compatibility for old interface."""
        return self.messages
    
    @property 
    def final_answer(self) -> Optional[Dict[str, str]]:
        """Backwards compatibility for final answer."""
        return self.get_final_answer()


# Alias for backwards compatibility
AgentMemory = SimpleMemory
EnhancedAgentMemory = SimpleMemory  # For easy migration


# Re-export commonly used classes
__all__ = [
    "SimpleMemory",
    "AgentMemory",
    "EnhancedAgentMemory",
    "AgentMessage",
    "ToolCallRecord", 
    "TaskState",
]