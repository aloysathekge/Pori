"""Data models for agent memory.

The Letta-style CoreMemory (named `Block`s always in the prompt), the
conversation/tool-call/task records (`AgentMessage`, `ToolCallRecord`,
`TaskState`), and the `SerializableMemoryState` handle used to rehydrate an
`AgentMemory` from its store.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

DEFAULT_CORE_BLOCK_LIMIT = 2000
LINE_NUMBER_REGEX = re.compile(r"\n\d+→ ")


@dataclass
class Block:
    label: str
    value: str = ""
    limit: int = DEFAULT_CORE_BLOCK_LIMIT
    read_only: bool = False

    def append(self, content: str) -> None:
        if self.read_only:
            raise ValueError(f"Block '{self.label}' is read-only")
        new_value = (self.value + "\n" + content).strip() if self.value else content
        self.value = (
            new_value[: self.limit] if len(new_value) > self.limit else new_value
        )

    def replace(self, old_string: str, new_string: str) -> None:
        if self.read_only:
            raise ValueError(f"Block '{self.label}' is read-only")
        if old_string not in self.value:
            raise ValueError(f"Text not found in block '{self.label}'")
        self.value = (self.value.replace(old_string, new_string))[: self.limit]

    def set_value(self, value: str) -> None:
        if self.read_only:
            raise ValueError(f"Block '{self.label}' is read-only")
        self.value = value[: self.limit] if len(value) > self.limit else value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "value": self.value,
            "limit": self.limit,
            "read_only": self.read_only,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Block":
        return cls(
            label=str(data.get("label", "")),
            value=str(data.get("value", "")),
            limit=int(data.get("limit", DEFAULT_CORE_BLOCK_LIMIT)),
            read_only=bool(data.get("read_only", False)),
        )


class CoreMemory:
    def __init__(self, block_limit: int = DEFAULT_CORE_BLOCK_LIMIT):
        self._blocks: Dict[str, Block] = {}
        self._block_limit = block_limit
        for label in ("persona", "human", "notes"):
            self._blocks[label] = Block(label=label, limit=block_limit)

    def get_block(self, label: str) -> Block:
        key = (label or "").strip()
        if not key:
            raise ValueError("Block label is required")
        if key not in self._blocks:
            self._blocks[key] = Block(label=key, limit=self._block_limit)
        return self._blocks[key]

    def update_block_value(self, label: str, value: str) -> None:
        block = self.get_block(label)
        block.set_value(value)

    def memory_insert(self, label: str, new_str: str, insert_line: int = -1) -> None:
        block = self.get_block(label)
        lines = block.value.splitlines()
        if insert_line < 0:
            idx = len(lines)
        else:
            idx = min(max(0, insert_line), len(lines))
        lines.insert(idx, new_str)
        new_value = "\n".join(lines).strip()
        if len(new_value) > block.limit:
            raise ValueError(
                f"Edit failed: New content ({len(new_value)} chars) exceeds block limit ({block.limit})"
            )
        block.set_value(new_value)

    def memory_rethink(self, label: str, new_memory: str) -> None:
        if LINE_NUMBER_REGEX.search(new_memory):
            raise ValueError("new_memory contains line-number prefixes")
        block = self.get_block(label)
        if len(new_memory) > block.limit:
            raise ValueError(
                f"Edit failed: New content ({len(new_memory)} chars) exceeds block limit ({block.limit})"
            )
        block.set_value(new_memory)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_limit": self._block_limit,
            "blocks": {label: block.to_dict() for label, block in self._blocks.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoreMemory":
        cm = cls(block_limit=int(data.get("block_limit", DEFAULT_CORE_BLOCK_LIMIT)))
        cm._blocks = {}
        for label, b in (data.get("blocks") or {}).items():
            cm._blocks[label] = Block.from_dict(b)
        for label in ("persona", "human", "notes"):
            if label not in cm._blocks:
                cm._blocks[label] = Block(label=label, limit=cm._block_limit)
        return cm

    def clone_read_only(self) -> "CoreMemory":
        """Return a deep copy with all blocks set to read-only."""
        cm = CoreMemory(block_limit=self._block_limit)
        cm._blocks = {}
        for label, block in self._blocks.items():
            cm._blocks[label] = Block(
                label=block.label,
                value=block.value,
                limit=block.limit,
                read_only=True,
            )
        return cm

    def compile(self) -> str:
        parts = []
        for label in ("persona", "human", "notes"):
            block = self._blocks.get(label)
            if block and block.value.strip():
                parts.append(f"<{label}>\n{block.value.strip()}\n</{label}>")
        if not parts:
            return ""
        return "<memory_blocks>\n" + "\n\n".join(parts) + "\n</memory_blocks>"


class AgentMessage(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ToolCallRecord(BaseModel):
    id: str = Field(default_factory=lambda: f"tool_{uuid.uuid4().hex[:12]}")
    tool_name: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    timestamp: datetime = Field(default_factory=datetime.now)
    task_id: Optional[str] = None
    # Write-ahead journal state: "dispatched" is persisted BEFORE the tool runs,
    # then flipped to "completed" with the real result. A record still marked
    # "dispatched" after a restart means the process died mid-tool — the call
    # may or may not have taken effect, so a resumed run must verify before
    # redoing it rather than blindly re-executing a side-effecting tool.
    status: str = "completed"


class TaskState(BaseModel):
    task_id: str
    description: str
    status: str = "in_progress"
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    # Loop-state checkpoint (persisted every step) so an interrupted run can be
    # resumed by a new Agent instead of restarting from step 0. `plan` mirrors
    # the model-owned PlanStore items as plain dicts.
    n_steps: int = 0
    consecutive_failures: int = 0
    current_activity: str = ""
    plan: List[Dict[str, Any]] = Field(default_factory=list)
    progress_updated_at: Optional[datetime] = None

    def complete(self, success: bool = True):
        self.status = "completed" if success else "failed"
        self.completed_at = datetime.now()


class SerializableMemoryState(BaseModel):
    namespace: str
    organization_id: str = "default_org"
    user_id: str
    agent_id: str
    session_id: str
    current_task_id: Optional[str] = None
    message_ids: List[str] = Field(default_factory=list)
    task_ids: List[str] = Field(default_factory=list)
    summary_count: int = 0
