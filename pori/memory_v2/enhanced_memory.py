from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .stores.in_memory import InMemoryStore
from .stores.vector_store import VectorMemoryStore


@dataclass
class ToolCallRecord:
    tool_name: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    timestamp: datetime


class EnhancedAgentMemory:
    """Façade providing working, long-term, and vector memory.

    Maintains backward-compatible fields/methods expected by current Agent.
    """

    def __init__(self, persistent: bool = False, vector: bool = True) -> None:
        # Working memory: short-term rolling buffer
        self.working = InMemoryStore()
        # Long-term memory: TODO swap to SqliteStore when persistent True
        self.long_term = InMemoryStore()
        # Vector memory for semantic recall
        self.vector = VectorMemoryStore() if vector else None

        # Back-compat structures expected by Agent
        self.tool_call_history: List[ToolCallRecord] = []
        self.tasks: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}
        self.summaries: List[Dict[str, Any]] = []
        self.final_answer: Optional[Dict[str, str]] = None

    # ---------------- Back-compat methods ----------------
    def add_message(self, role: str, content: str) -> None:
        key = f"msg_{len(self.working.data)}"
        self.working.add(key, f"{role}: {content}")

    def get_recent_messages_structured(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return last n messages as structured dicts with role/content.

        This provides compatibility for agents expecting structured messages.
        """
        values = list(self.working.data.values())[-n:]
        structured: List[Dict[str, Any]] = []
        for v in values:
            # Expect format "role: content"; fall back to system if parse fails
            if ": " in v:
                role, content = v.split(": ", 1)
            else:
                role, content = "system", v
            structured.append({"role": role, "content": content})
        return structured

    def add_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Dict[str, Any],
        success: bool,
    ) -> None:
        rec = ToolCallRecord(
            tool_name=tool_name,
            parameters=parameters,
            result=result,
            success=success,
            timestamp=datetime.now(),
        )
        self.tool_call_history.append(rec)

    def update_state(self, key: str, value: Any) -> None:
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def get_final_answer(self) -> Optional[Dict[str, str]]:
        """Compatibility with legacy memory API."""
        return self.get_state("final_answer")

    def create_task(self, task_id: str, description: str):
        """Create a TaskState using the legacy class to preserve behavior."""
        try:
            from ..memory import TaskState  # lazy import to avoid circulars
        except Exception:
            # Fallback minimal structure if import fails
            class _Task:
                def __init__(self, task_id: str, description: str):
                    self.task_id = task_id
                    self.description = description
                    self.status = "in_progress"

                def complete(self, success: bool = True):
                    self.status = "completed" if success else "failed"

            task = _Task(task_id, description)
            self.tasks[task_id] = task
            return task

        task = TaskState(task_id=task_id, description=description)
        self.tasks[task_id] = task
        return task

    def create_summary(self, step: int, max_messages: int = 10) -> str:
        # Convert last N working messages into textual summary and store long-term
        recent = list(self.working.data.values())[-max_messages:]
        summary_text = "\n".join(recent)
        meta = {
            "type": "summary",
            "step": step,
            "created_at": datetime.now().isoformat(),
        }
        key = f"summary_{len(self.summaries)}"
        self.long_term.add(key, summary_text, meta=meta)
        if self.vector and summary_text:
            # Also index summary text for semantic recall
            self.vector.add(key, summary_text)
        self.summaries.append({"key": key, **meta})
        return f"Step {step} summary stored as {key} with {len(recent)} messages"

    # ---------------- Enhanced APIs ----------------
    def add_experience(self, text: str, importance: int = 1) -> str:
        key = f"exp_{len(self.long_term.data)}"
        self.long_term.add(key, text, meta={"importance": importance})
        if self.vector:
            self.vector.add(key, text)
        return key

    def recall(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        if self.vector:
            return self.vector.search(query, k)
        return self.long_term.search(query, k)

    def get_recent_messages(self, n: int = 10) -> str:
        values = list(self.working.data.values())[-n:]
        return "\n".join(values)
