"""Persistent agent memory (Letta-style CoreMemory blocks + session records).

Package split for readability, same public surface as the old ``pori.memory``
module: data models in `models`, pluggable persistence backends in `stores`,
and the `AgentMemory` facade in `agent_memory` — everything is re-exported
here so ``from pori.memory import X`` keeps working (including the
``pori.memory_stores`` entry-point factories referenced in pyproject.toml).
"""

from __future__ import annotations

from ..memory_contracts import (
    ConflictPolicy,
    MemoryCatalog,
    MemoryHit,
    MemoryKind,
    MemoryProvenance,
    MemoryRecord,
    MemoryRetention,
    MemoryScope,
    MemorySensitivity,
)
from .agent_memory import AgentMemory
from .models import (
    DEFAULT_CORE_BLOCK_LIMIT,
    LINE_NUMBER_REGEX,
    AgentMessage,
    Block,
    CoreMemory,
    SerializableMemoryState,
    TaskState,
    ToolCallRecord,
)
from .stores import (
    InMemoryMemoryStore,
    MemoryStore,
    SQLiteMemoryStore,
    create_in_memory_store,
    create_memory_store,
    create_sqlite_memory_store,
)

__all__ = [
    "AgentMemory",
    "AgentMessage",
    "ToolCallRecord",
    "TaskState",
    "Block",
    "CoreMemory",
    "SerializableMemoryState",
    "MemoryStore",
    "InMemoryMemoryStore",
    "SQLiteMemoryStore",
    "create_memory_store",
    "create_in_memory_store",
    "create_sqlite_memory_store",
    "ConflictPolicy",
    "MemoryCatalog",
    "MemoryHit",
    "MemoryKind",
    "MemoryProvenance",
    "MemoryRecord",
    "MemoryRetention",
    "MemoryScope",
    "MemorySensitivity",
]
