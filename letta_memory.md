# Letta Memory System - Complete Implementation Guide

> **Based on Letta OSS (formerly MemGPT)**  
> A comprehensive guide to implementing Letta's innovative memory architecture in your own AI agent

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [How Letta's Memory System Works (Codebase Deep Dive)](#how-lettas-memory-system-works-codebase-deep-dive)
4. [Core Data Models](#core-data-models)
5. [Agent State Management](#agent-state-management)
6. [Memory Compilation & Rendering](#memory-compilation--rendering)
7. [Memory Editing Tools](#memory-editing-tools)
8. [Message Management](#message-management)
9. [Context Window Management](#context-window-management)
10. [Archival Memory (Vector Search)](#archival-memory-vector-search)
11. [Agent Integration](#agent-integration)
12. [Streaming Support](#streaming-support)
13. [Tool Rules System](#tool-rules-system)
14. [Persistence Layer](#persistence-layer)
15. [Error Handling & Recovery](#error-handling--recovery)
16. [Complete Working Example](#complete-working-example)
17. [Testing](#testing)
18. [Production Deployment](#production-deployment)
19. [Best Practices](#best-practices)
20. [How to Replicate in Your Agent Framework](#how-to-replicate-in-your-agent-framework)

---

## Overview

### The Three-Tier Memory Architecture

Letta implements a hierarchical memory system inspired by computer operating systems:

```
┌─────────────────────────────────────────────────────────────┐
│                 CORE MEMORY (In-Context)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Persona    │  │    Human     │  │   [Custom]   │      │
│  │    Block     │  │    Block     │  │    Blocks    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  Always visible to LLM | Editable via tools | Size-limited │
└─────────────────────────────────────────────────────────────┘
           ▲                                    ▲
           │ Rebuild on every step              │ Search/Retrieve
           │                                    │
┌──────────┴────────────────────────────────────┴─────────────┐
│                    AGENT STATE                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Block IDs  │  │ Message IDs │  │ Tool Configuration  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  Persisted | Serializable | Separate from stateful Agent    │
└─────────────────────────────────────────────────────────────┘
           ▲
           │ Search/Summarize
           ▼
┌─────────────────────────────────────────────────────────────┐
│              RECALL MEMORY (Conversation History)           │
│  Full message history | Semantic + text search | Summaries  │
└─────────────────────────────────────────────────────────────┘
           ▲
           │ Search/Retrieve
           ▼
┌─────────────────────────────────────────────────────────────┐
│            ARCHIVAL MEMORY (Long-term Storage)              │
│  Semantic embeddings | Vector search | Unlimited capacity   │
└─────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Self-Editing Memory**: Agents modify their own memory via tool calls
2. **State Separation**: Stateful `Agent` is separate from serializable `AgentState`
3. **Always In-Context**: Core memory is part of every system prompt
4. **Structured Format**: XML-based format for clarity
5. **Size Limits**: Character limits prevent unbounded growth
6. **Automatic Summarization**: Context overflow triggers automatic summarization
7. **Persistence**: All changes saved to database immediately

---

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT REQUEST                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                AGENT                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  Agent State    │  │  Memory Object  │  │  Tool Registry              │  │
│  │  - block_ids    │  │  - blocks[]     │  │  - core_memory_append       │  │
│  │  - message_ids  │  │  - file_blocks[]│  │  - core_memory_replace      │  │
│  │  - tool_ids     │  │  - compile()    │  │  - archival_memory_insert   │  │
│  │  - llm_config   │  │                 │  │  - conversation_search      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
            │                    │                         │
            ▼                    ▼                         ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────────────────┐
│  BlockManager     │ │  MessageManager   │ │  ContextWindowManager         │
│  - save_block()   │ │  - create()       │ │  - count_tokens()             │
│  - load_block()   │ │  - get_by_ids()   │ │  - check_overflow()           │
│  - update_value() │ │  - append()       │ │  - summarize_inplace()        │
└───────────────────┘ └───────────────────┘ └───────────────────────────────┘
            │                    │                         │
            └────────────────────┼─────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATABASE LAYER                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   blocks    │  │  messages   │  │   agents    │  │  archival_memory    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Agent Step

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. CHECK CONTEXT WINDOW                                                      │
│    - Count tokens in current context                                         │
│    - If >75% full: trigger automatic summarization                           │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. REBUILD SYSTEM PROMPT                                                     │
│    - Load latest block values from database                                  │
│    - Compile memory into XML format                                          │
│    - Add memory statistics metadata                                          │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. CALL LLM                                                                  │
│    - System prompt + in-context messages + tools                             │
│    - Get response (may include tool calls)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. EXECUTE TOOLS (if any)                                                    │
│    - Memory tools: Update blocks, persist to DB                              │
│    - Archival tools: Insert/search vector storage                            │
│    - Check for heartbeat request (continue chaining)                         │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. PERSIST & RETURN                                                          │
│    - Save all new messages to database                                       │
│    - Update agent state (message_ids)                                        │
│    - Return response to user                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## How Letta's Memory System Works (Codebase Deep Dive)

This section traces the **exact** flow of memory in the Letta codebase, with file and function references so you can verify behavior or port it faithfully.

### 1. Where Memory Lives

| Concept | Where It Lives | Key Types / Functions |
|--------|----------------|------------------------|
| **Core memory (blocks)** | In-memory: `AgentState.memory` (a `Memory` with `blocks: List[Block]`). Persisted: `Block` ORM → `block_manager` | `letta/schemas/memory.py` (`Memory`, `compile()`), `letta/schemas/block.py` (`Block`) |
| **In-context messages** | Agent holds **message IDs** only: `AgentState.message_ids`. First ID = system message; rest = conversation in order. Messages are fetched by ID. | `AgentState.message_ids` in `letta/schemas/agent.py`; `get_in_context_messages` in `letta/services/agent_manager.py` |
| **System message content** | A **message** in the same store as conversation messages; its content is the full system prompt (base prompt + compiled memory + metadata). Rebuilt when core memory changes. | `agent_manager.rebuild_system_prompt_async`; `compile_system_message` in `letta/services/helpers/agent_manager_helper.py` |
| **Archival memory** | Passages with embeddings; search/insert via tools. | `letta/services/passage_manager.py`; `archival_memory_search` / `archival_memory_insert` in `letta/services/tool_executor/core_tool_executor.py` |

### 2. Building the System Prompt (Step-by-Step)

1. **Base system prompt**  
   Stored on the agent as `AgentState.system` (string). It may contain the placeholder `{CORE_MEMORY}` (see `IN_CONTEXT_MEMORY_KEYWORD` in `letta/constants.py`).

2. **Compile in-context memory to a string**  
   - Entry: `agent_state.memory.compile(...)` in `letta/schemas/memory.py`.  
   - Renders:
     - **Memory blocks** (for non–react/workflow agents): XML `<memory_blocks>`, then for each block: `<label>`, `<description>`, `<metadata>`, `<value>` (and optionally line-numbered value for some agent types + Anthropic).  
     - **Tool usage rules** (if provided): `<tool_usage_rules>`.  
     - **Sources/directories** (if provided): `<directories>` with file blocks.  
   - Returns a single string (no system prompt yet).

3. **Memory metadata block**  
   `PromptGenerator.compile_memory_metadata_block()` in `letta/prompts/prompt_generator.py` builds the `<memory_metadata>` section: current time, last memory edit time, count of messages in recall, count (and tags) of archival memories. This tells the agent “how much is in recall/archival” so it knows to use tools.

4. **Full system message string**  
   - **Sync path**: `compile_system_message()` in `letta/services/helpers/agent_manager_helper.py`.  
   - **Async path**: `PromptGenerator.get_system_message_from_compiled_memory()` in `letta/prompts/prompt_generator.py`.  
   - They:
     - Take `memory_with_sources` = output of `memory.compile(...)`.
     - Append `memory_metadata_string`.
     - Replace `{CORE_MEMORY}` in `agent_state.system` with that combined string (or append it if the placeholder is missing and `append_icm_if_missing` is True).  
   - Result = the full system message text sent to the LLM.

5. **When the system message is rebuilt**  
   - **After core memory edits**: Every core memory tool (e.g. `core_memory_append`, `core_memory_replace`) calls `agent_manager.update_memory_if_changed_async()`. That function:
     - Compares new compiled memory string to current system message content.
     - If different: persists block value changes via `block_manager.update_block_async()`, reloads blocks into `agent_state.memory`, then calls `rebuild_system_prompt_async()`.
   - **Rebuild** (`rebuild_system_prompt_async` in `letta/services/agent_manager.py`):
     - Loads agent state (with memory, sources, tools).
     - Gets current system message (first message in `message_ids`).
     - Compiles current memory (with tool rules, sources, etc.); if that string is already contained in the current system message and not `force`, skip.
     - Builds new system message string via `PromptGenerator.get_system_message_from_compiled_memory(...)`.
     - If there’s a diff: updates that message in place via `message_manager.update_message_by_id_async()` (same message ID; content replaced).  
   So: **core memory change → block DB update → system message content refresh (same message ID)**.

### 3. In-Context Messages (Recall / Conversation)

- **Storage**: Messages are stored in the message store; the agent only keeps a list of **message IDs**: `agent_state.message_ids`.  
- **Order**: `message_ids[0]` is the system message; the rest are conversation (user/assistant/tool) in chronological order.  
- **Loading**: `agent_manager.get_in_context_messages(agent_id, actor)` → `message_manager.get_messages_by_ids_async(agent_state.message_ids, actor)` returns the actual message objects in ID order.  
- **Appending**: New messages are created with `message_manager.create_many_messages_async()`; their IDs are appended to `agent.message_ids`, and the agent is updated via `set_in_context_messages_async`.  
- **Trimming**: e.g. `trim_older_in_context_messages(num, agent_id, actor)` keeps `message_ids[0]` (system) and drops the next `num` IDs, then saves the new list. Used after summarization.  
- **Summarization (legacy agent path)**: When context overflows, `summarize_messages_inplace()` (in `letta/agent.py`) computes a cutoff, summarizes messages `[1:cutoff]` via `summarize_messages()`, then trims those from in-context and prepends a single “summary” user message. (The actual `summarize_messages` implementation in `letta/memory.py` is currently commented out; newer agents use `letta/services/summarizer/`.)

### 4. Core Memory Tools (Append / Replace)

- **Execution**: `LettaCoreToolExecutor` in `letta/services/tool_executor/core_tool_executor.py` maps tool names to methods (e.g. `core_memory_append`, `core_memory_replace`).  
- **Logic**:  
  - **core_memory_append(agent_state, actor, label, content)**: Read current block value with `agent_state.memory.get_block(label).value`, append `"\n" + content`, call `agent_state.memory.update_block_value(label, new_value)`, then `await agent_manager.update_memory_if_changed_async(agent_id, agent_state.memory, actor)`.  
  - **core_memory_replace(agent_state, actor, label, old_content, new_content)**: Same get/update on block; replace `old_content` with `new_content` (exact match required); then `update_memory_if_changed_async`.  
- **Persistence**: `update_memory_if_changed_async` (in `letta/services/agent_manager.py`) writes only **changed** block values to the DB via `block_manager.update_block_async`, then refreshes `agent_state.memory` from the DB and calls `rebuild_system_prompt_async` so the system message content stays in sync.

### 5. Archival Memory (Vector Search)

- **Insert**: `archival_memory_insert` in core tool executor calls `passage_manager.insert_passage(agent_state, text, actor, tags)` then `agent_manager.rebuild_system_prompt_async(..., force=True)` so the metadata line “X memories in archival” updates.  
- **Search**: `archival_memory_search` calls `agent_manager.search_agent_archival_memory_async(...)` (query, tags, filters); results are formatted and returned to the agent.  
- **Storage**: Passages (with embeddings) are stored and queried via the passage/archive ORM and vector DB (e.g. pgvector). See `letta/services/passage_manager.py` and archival passage models.

### 6. Constants and Keywords

- **`IN_CONTEXT_MEMORY_KEYWORD`** = `"CORE_MEMORY"` in `letta/constants.py`. The system prompt template uses `{CORE_MEMORY}`; it is replaced by the compiled memory + metadata.  
- **Block limits**: `CORE_MEMORY_BLOCK_CHAR_LIMIT`, `CORE_MEMORY_PERSONA_CHAR_LIMIT`, `CORE_MEMORY_HUMAN_CHAR_LIMIT` in `letta/constants.py`.  
- **Summarization**: `SUMMARIZATION_TRIGGER_MULTIPLIER` (e.g. 1.0 = trigger when usage exceeds context window). Context overflow is detected (e.g. by LLM error or token counting) and triggers summarization or trimming depending on agent type.

### 7. End-to-End Flow (Single Step with Memory Edit)

1. Request comes in; agent state loaded (including `memory` = list of blocks, `message_ids`).  
2. System prompt is built (or reused): compile memory → add metadata → substitute `{CORE_MEMORY}` in `agent_state.system`.  
3. In-context messages = `get_in_context_messages()` → messages for `message_ids` in order.  
4. LLM is called with system message + in-context messages + tools.  
5. If the model calls e.g. `core_memory_append`: executor updates `agent_state.memory` in place, then `update_memory_if_changed_async` persists block changes and rebuilds the system message (same message ID, new content).  
6. New assistant (and tool) messages are created and appended to `message_ids`.  
7. Response returned. Next step will load the same `message_ids` (now including the new messages) and the updated system message (with new memory content).

---

## Core Data Models

### 1. Block - The Fundamental Memory Unit

```python
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
import uuid

@dataclass
class Block:
    """
    A Block represents a reserved section of the agent's core memory.
    Each block has a label, value, and metadata.
    
    In Letta, blocks are the fundamental unit of in-context memory.
    They are always visible to the LLM and editable via tools.
    """
    # Core fields
    id: str = field(default_factory=lambda: f"block-{uuid.uuid4()}")
    label: str = ""                      # e.g., "persona", "human", "notes"
    value: str = ""                      # The actual memory content
    
    # Metadata
    description: Optional[str] = None    # What this block is for (shown to LLM)
    limit: int = 2000                    # Character limit
    read_only: bool = False              # Can agent edit this?
    
    # Organization fields
    is_template: bool = False
    template_name: Optional[str] = None
    project_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Audit fields
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by_id: Optional[str] = None
    last_updated_by_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate block on creation"""
        self.validate()
    
    def validate(self):
        """Validate block constraints"""
        if not self.label:
            raise ValueError("Block must have a label")
        
        if len(self.value) > self.limit:
            raise ValueError(
                f"Edit failed: Exceeds {self.limit} character limit "
                f"(requested {len(self.value)} characters)"
            )
    
    def __len__(self) -> int:
        """Return character count of value"""
        return len(self.value)
    
    def __setattr__(self, name, value):
        """Run validation when value is updated"""
        super().__setattr__(name, value)
        if name == "value" and hasattr(self, "limit"):
            if len(value) > self.limit:
                raise ValueError(
                    f"Edit failed: Exceeds {self.limit} character limit "
                    f"(requested {len(value)} characters)"
                )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "label": self.label,
            "value": self.value,
            "description": self.description,
            "limit": self.limit,
            "read_only": self.read_only,
            "is_template": self.is_template,
            "template_name": self.template_name,
            "project_id": self.project_id,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by_id": self.created_by_id,
            "last_updated_by_id": self.last_updated_by_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Block':
        """Create block from dictionary"""
        data = data.copy()
        # Handle datetime fields
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        # Handle legacy field names
        if "char_limit" in data:
            data["limit"] = data.pop("char_limit")
        return cls(**data)


# Specialized block types
@dataclass
class PersonaBlock(Block):
    """Block for agent persona/personality"""
    label: str = "persona"
    description: str = "Information about your personality and behavior"


@dataclass  
class HumanBlock(Block):
    """Block for information about the user"""
    label: str = "human"
    description: str = "Information about the user you are conversing with"


@dataclass
class FileBlock(Block):
    """
    Special block type for representing file content in agent's memory.
    Used when agents have access to a file system.
    """
    file_id: str = ""
    source_id: str = ""                  # Folder/directory ID
    is_open: bool = False                # Is file currently open?
    last_accessed_at: Optional[datetime] = None
    
    def mark_accessed(self):
        """Update last accessed timestamp"""
        self.last_accessed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
```

### 2. Memory - The Container

```python
from typing import List, Dict, Optional, Union
from io import StringIO
from enum import Enum

class AgentType(str, Enum):
    """Agent types that affect memory rendering"""
    MEMGPT_AGENT = "memgpt_agent"
    LETTA_V1_AGENT = "letta_v1_agent"
    REACT_AGENT = "react_agent"
    SLEEPTIME_AGENT = "sleeptime_agent"
    VOICE_AGENT = "voice_convo_agent"


class Memory:
    """
    Represents the in-context memory (Core Memory) of the agent.
    Contains multiple Block objects that are always visible to the LLM.
    
    Key responsibilities:
    - Store and manage memory blocks
    - Compile blocks into prompt format
    - Validate block constraints
    """
    
    def __init__(
        self, 
        blocks: List[Block] = None,
        file_blocks: List[FileBlock] = None,
        agent_type: Optional[AgentType] = None
    ):
        self.blocks: List[Block] = blocks or []
        self.file_blocks: List[FileBlock] = file_blocks or []
        self.agent_type = agent_type
        self._validate_no_duplicate_labels()
    
    def _validate_no_duplicate_labels(self):
        """Ensure no duplicate block labels"""
        labels = [b.label for b in self.blocks]
        if len(labels) != len(set(labels)):
            duplicates = [l for l in labels if labels.count(l) > 1]
            raise ValueError(f"Duplicate block labels found: {set(duplicates)}")
    
    def get_block(self, label: str) -> Block:
        """
        Get a block by its label.
        
        Args:
            label: The block label to search for
            
        Returns:
            The Block object
            
        Raises:
            KeyError: If block not found
        """
        for block in self.blocks:
            if block.label == label:
                return block
        
        available = [b.label for b in self.blocks]
        raise KeyError(
            f"Block field '{label}' does not exist "
            f"(available sections = {', '.join(available)})"
        )
    
    def get_blocks(self) -> List[Block]:
        """Return all memory blocks"""
        return self.blocks
    
    def list_block_labels(self) -> List[str]:
        """Return list of all block labels"""
        return [block.label for block in self.blocks]
    
    def set_block(self, block: Block):
        """
        Add or update a block.
        If a block with the same label exists, it replaces it.
        Otherwise, appends the new block.
        """
        block.validate()
        for i, existing_block in enumerate(self.blocks):
            if existing_block.label == block.label:
                self.blocks[i] = block
                return
        self.blocks.append(block)
    
    def update_block_value(self, label: str, value: str):
        """
        Update the value of a specific block.
        
        Args:
            label: Block label to update
            value: New value
            
        Raises:
            ValueError: If value exceeds block limit or block is read-only
            KeyError: If block not found
        """
        if not isinstance(value, str):
            raise ValueError("Provided value must be a string")
        
        block = self.get_block(label)
        
        if block.read_only:
            raise ValueError(
                f"Cannot modify read-only block '{label}'. "
                "Read-only blocks can only be edited by system processes."
            )
        
        if len(value) > block.limit:
            raise ValueError(
                f"Edit failed: Exceeds {block.limit} character limit "
                f"(requested {len(value)} characters)"
            )
        
        block.value = value
        block.updated_at = datetime.utcnow()
    
    def remove_block(self, label: str):
        """Remove a block by label"""
        self.blocks = [b for b in self.blocks if b.label != label]
    
    def total_characters(self) -> int:
        """Return total character count across all blocks"""
        return sum(len(block) for block in self.blocks)
    
    def to_dict(self) -> dict:
        """Serialize memory to dictionary"""
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "file_blocks": [fb.to_dict() for fb in self.file_blocks],
            "agent_type": self.agent_type.value if self.agent_type else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Memory':
        """Deserialize memory from dictionary"""
        blocks = [Block.from_dict(b) for b in data.get("blocks", [])]
        file_blocks = [FileBlock.from_dict(fb) for fb in data.get("file_blocks", [])]
        agent_type = AgentType(data["agent_type"]) if data.get("agent_type") else None
        return cls(blocks=blocks, file_blocks=file_blocks, agent_type=agent_type)
```

---

## Agent State Management

### AgentState - The Serializable State

This is **critical** and often missed. Letta separates the stateful `Agent` object from the serializable `AgentState`:

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import copy

@dataclass
class LLMConfig:
    """Configuration for the LLM"""
    model: str = "gpt-4o"
    model_endpoint_type: str = "openai"
    model_endpoint: Optional[str] = None
    context_window: int = 128000
    max_tokens: Optional[int] = None
    temperature: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "model_endpoint_type": self.model_endpoint_type,
            "model_endpoint": self.model_endpoint,
            "context_window": self.context_window,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'LLMConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings"""
    embedding_model: str = "text-embedding-3-small"
    embedding_endpoint_type: str = "openai"
    embedding_dim: int = 1536
    embedding_chunk_size: int = 300
    
    def to_dict(self) -> dict:
        return {
            "embedding_model": self.embedding_model,
            "embedding_endpoint_type": self.embedding_endpoint_type,
            "embedding_dim": self.embedding_dim,
            "embedding_chunk_size": self.embedding_chunk_size,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EmbeddingConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AgentState:
    """
    Complete agent state - this is what gets persisted and loaded.
    
    IMPORTANT: Letta separates the stateful Agent object from the 
    serializable AgentState. The Agent holds runtime state (LLM client,
    connections), while AgentState holds all persistable configuration.
    
    This separation allows:
    - Clean serialization/deserialization
    - Multiple Agent instances sharing state
    - Easy state comparison and versioning
    """
    # Identity
    id: str = field(default_factory=lambda: f"agent-{uuid.uuid4()}")
    name: str = "Agent"
    description: Optional[str] = None
    created_by_id: str = "user-default"
    
    # Memory block IDs (not the blocks themselves!)
    block_ids: List[str] = field(default_factory=list)
    
    # In-context message IDs (not the messages themselves!)
    message_ids: List[str] = field(default_factory=list)
    
    # Tool configuration
    tool_ids: List[str] = field(default_factory=list)
    tool_rules: Optional[List[Dict]] = None
    
    # System prompt
    system: str = ""  # The base system instructions
    
    # Agent type
    agent_type: str = "letta_v1_agent"
    
    # Model configuration
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    
    # Behavior flags
    message_buffer_autoclear: bool = False  # Clear messages after each response
    enable_sleeptime: bool = False  # Background memory management
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Runtime tracking (not persisted to main state)
    _memory: Optional[Memory] = field(default=None, repr=False)
    
    @property
    def memory(self) -> Optional[Memory]:
        """Get the memory object (must be loaded separately)"""
        return self._memory
    
    @memory.setter
    def memory(self, value: Memory):
        """Set the memory object"""
        self._memory = value
    
    def __deepcopy__(self, memo=None) -> 'AgentState':
        """Allow safe copying for tool execution"""
        return AgentState(
            id=self.id,
            name=self.name,
            description=self.description,
            created_by_id=self.created_by_id,
            block_ids=copy.copy(self.block_ids),
            message_ids=copy.copy(self.message_ids),
            tool_ids=copy.copy(self.tool_ids),
            tool_rules=copy.deepcopy(self.tool_rules) if self.tool_rules else None,
            system=self.system,
            agent_type=self.agent_type,
            llm_config=copy.deepcopy(self.llm_config),
            embedding_config=copy.deepcopy(self.embedding_config),
            message_buffer_autoclear=self.message_buffer_autoclear,
            enable_sleeptime=self.enable_sleeptime,
            metadata=copy.deepcopy(self.metadata) if self.metadata else None,
            tags=copy.copy(self.tags),
            created_at=self.created_at,
            updated_at=self.updated_at,
            _memory=copy.deepcopy(self._memory) if self._memory else None,
        )
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_by_id": self.created_by_id,
            "block_ids": self.block_ids,
            "message_ids": self.message_ids,
            "tool_ids": self.tool_ids,
            "tool_rules": self.tool_rules,
            "system": self.system,
            "agent_type": self.agent_type,
            "llm_config": self.llm_config.to_dict() if self.llm_config else None,
            "embedding_config": self.embedding_config.to_dict() if self.embedding_config else None,
            "message_buffer_autoclear": self.message_buffer_autoclear,
            "enable_sleeptime": self.enable_sleeptime,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AgentState':
        """Deserialize from dictionary"""
        data = data.copy()
        
        # Handle datetime fields
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        
        # Handle nested configs
        if data.get("llm_config") and isinstance(data["llm_config"], dict):
            data["llm_config"] = LLMConfig.from_dict(data["llm_config"])
        if data.get("embedding_config") and isinstance(data["embedding_config"], dict):
            data["embedding_config"] = EmbeddingConfig.from_dict(data["embedding_config"])
        
        # Remove non-init fields
        data.pop("_memory", None)
        
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
```

---

## Memory Compilation & Rendering

### The Core: Rendering Memory as Prompt

```python
class Memory:
    """Memory class continued with compile methods..."""
    
    def compile(
        self, 
        include_line_numbers: bool = False,
        sources: Optional[List] = None,
        max_files_open: Optional[int] = None,
        llm_config: Optional[LLMConfig] = None
    ) -> str:
        """
        Compile memory blocks into a formatted string for the system prompt.
        This is the key method that makes memory visible to the LLM.
        
        Args:
            include_line_numbers: Add line numbers for precise editing
            sources: File sources/folders attached to agent
            max_files_open: Maximum number of files that can be open
            llm_config: LLM configuration (affects rendering)
            
        Returns:
            Formatted XML string containing all memory blocks
        """
        s = StringIO()
        
        # Determine rendering style based on agent type and model
        is_line_numbered = False
        if llm_config and hasattr(llm_config, 'model_endpoint_type'):
            is_anthropic = llm_config.model_endpoint_type == "anthropic"
            is_line_numbered_type = self.agent_type in (
                AgentType.SLEEPTIME_AGENT,
                AgentType.LETTA_V1_AGENT,
            )
            is_line_numbered = is_line_numbered_type and is_anthropic
        
        if include_line_numbers:
            is_line_numbered = True
        
        # Render based on agent type
        is_react = self.agent_type in (AgentType.REACT_AGENT,)
        
        if not is_react:
            if is_line_numbered:
                self._render_memory_blocks_line_numbered(s)
            else:
                self._render_memory_blocks_standard(s)
        
        # Render file system if sources provided
        if sources:
            self._render_directories(s, sources, max_files_open)
        
        return s.getvalue()
    
    def _render_memory_blocks_standard(self, s: StringIO):
        """Render memory blocks in standard XML format"""
        if not self.blocks:
            s.write("")
            return
        
        s.write("<memory_blocks>\n")
        s.write("The following memory blocks are currently engaged in your core memory unit:\n\n")
        
        for idx, block in enumerate(self.blocks):
            label = block.label or "block"
            value = block.value or ""
            desc = block.description or ""
            chars_current = len(value)
            limit = block.limit
            
            s.write(f"<{label}>\n")
            s.write("<description>\n")
            s.write(f"{desc}\n")
            s.write("</description>\n")
            s.write("<metadata>")
            
            if block.read_only:
                s.write("\n- read_only=true")
            
            s.write(f"\n- chars_current={chars_current}")
            s.write(f"\n- chars_limit={limit}\n")
            s.write("</metadata>\n")
            s.write("<value>\n")
            s.write(f"{value}\n")
            s.write("</value>\n")
            s.write(f"</{label}>\n")
            
            if idx != len(self.blocks) - 1:
                s.write("\n")
        
        s.write("\n</memory_blocks>")
    
    def _render_memory_blocks_line_numbered(self, s: StringIO):
        """
        Render memory blocks with line numbers for precise editing.
        Used with Claude/Anthropic models and sleeptime agents.
        """
        LINE_NUMBER_WARNING = (
            "NOTE: Line numbers shown below (with arrows like '1→') are "
            "provided to help you locate text during editing. Do NOT include "
            "line number prefixes in your memory edit tool calls."
        )
        
        if not self.blocks:
            s.write("")
            return
        
        s.write("<memory_blocks>\n")
        s.write("The following memory blocks are currently engaged in your core memory unit:\n\n")
        
        for idx, block in enumerate(self.blocks):
            label = block.label or "block"
            value = block.value or ""
            desc = block.description or ""
            limit = block.limit
            
            s.write(f"<{label}>\n")
            s.write("<description>\n")
            s.write(f"{desc}\n")
            s.write("</description>\n")
            s.write("<metadata>")
            
            if block.read_only:
                s.write("\n- read_only=true")
            
            s.write(f"\n- chars_current={len(value)}")
            s.write(f"\n- chars_limit={limit}\n")
            s.write("</metadata>\n")
            s.write(f"<warning>\n{LINE_NUMBER_WARNING}\n</warning>\n")
            s.write("<value>\n")
            
            if value:
                # Add line numbers
                for i, line in enumerate(value.split("\n"), start=1):
                    s.write(f"{i}→ {line}\n")
            
            s.write("</value>\n")
            s.write(f"</{label}>\n")
            
            if idx != len(self.blocks) - 1:
                s.write("\n")
        
        s.write("\n</memory_blocks>")
    
    def _render_directories(
        self, 
        s: StringIO, 
        sources: List, 
        max_files_open: Optional[int]
    ):
        """
        Render attached file system directories.
        Shows which files are open and their content.
        """
        s.write("\n\n<directories>\n")
        
        if max_files_open is not None:
            current_open = sum(1 for fb in self.file_blocks if fb.is_open)
            s.write("<file_limits>\n")
            s.write(f"- current_files_open={current_open}\n")
            s.write(f"- max_files_open={max_files_open}\n")
            s.write("</file_limits>\n")
        
        for source in sources:
            source_name = getattr(source, "name", "")
            source_desc = getattr(source, "description", None)
            source_instr = getattr(source, "instructions", None)
            source_id = getattr(source, "id", None)
            
            s.write(f'<directory name="{source_name}">\n')
            
            if source_desc:
                s.write(f"<description>{source_desc}</description>\n")
            if source_instr:
                s.write(f"<instructions>{source_instr}</instructions>\n")
            
            # List files in this directory
            for fb in self.file_blocks:
                if fb.source_id == source_id:
                    status = "open" if fb.is_open else "closed"
                    label = fb.label
                    desc = fb.description or ""
                    
                    s.write(f'<file status="{status}" name="{label}">\n')
                    
                    if desc:
                        s.write("<description>\n")
                        s.write(f"{desc}\n")
                        s.write("</description>\n")
                    
                    s.write("<metadata>")
                    if fb.read_only:
                        s.write("\n- read_only=true")
                    s.write(f"\n- chars_current={len(fb.value or '')}")
                    s.write(f"\n- chars_limit={fb.limit}\n")
                    s.write("</metadata>\n")
                    
                    if fb.is_open and fb.value:
                        s.write("<value>\n")
                        s.write(f"{fb.value}\n")
                        s.write("</value>\n")
                    
                    s.write("</file>\n")
            
            s.write("</directory>\n")
        
        s.write("</directories>")


# Example output of compile():
EXAMPLE_MEMORY_OUTPUT = """
<memory_blocks>
The following memory blocks are currently engaged in your core memory unit:

<persona>
<description>
Information about your personality and behavior as an AI assistant
</description>
<metadata>
- chars_current=156
- chars_limit=2000
</metadata>
<value>
I am a helpful AI assistant focused on teaching programming concepts.
I prefer to provide concrete code examples alongside explanations.
I'm patient and encourage learning through experimentation.
</value>
</persona>

<human>
<description>
Information about the user you are conversing with
</description>
<metadata>
- chars_current=89
- chars_limit=2000
</metadata>
<value>
User's name is Alice.
She is a software engineer learning about AI agent architectures.
She prefers Python examples with detailed comments.
</value>
</human>

</memory_blocks>
"""
```

---

## Memory Editing Tools

### Tool Definitions for LLM Function Calling

```python
from typing import Optional, List, Literal
import re
import json

# Constants
LINE_NUMBER_REGEX = re.compile(r"\n\d+→ ")
REQUEST_HEARTBEAT_PARAM = "request_heartbeat"


class MemoryTools:
    """
    Collection of tools that allow the agent to edit its own memory.
    These are exposed to the LLM via function calling.
    
    The agent calls these tools to:
    - Append new information to memory
    - Replace existing information
    - Completely rewrite memory blocks
    - Search archival and conversation memory
    """
    
    def __init__(
        self, 
        memory: Memory,
        archival_manager: Optional['ArchivalMemoryManager'] = None,
        message_manager: Optional['MessageManager'] = None,
        agent_id: str = "default"
    ):
        self.memory = memory
        self.archival_manager = archival_manager
        self.message_manager = message_manager
        self.agent_id = agent_id
    
    # ===== CORE MEMORY TOOLS =====
    
    def core_memory_append(self, label: str, content: str) -> Optional[str]:
        """
        Append text to the end of a memory block.
        
        Args:
            label: The memory block to edit (e.g., "human", "persona")
            content: Text to append to the block. All unicode (including emojis) supported.
            
        Returns:
            None (no message returned to LLM)
            
        Example:
            core_memory_append(label="human", content="Alice prefers dark mode.")
        """
        block = self.memory.get_block(label)
        
        if block.read_only:
            raise ValueError(
                f"Block '{label}' is read-only and cannot be edited. "
                "Only system processes can modify read-only blocks."
            )
        
        current_value = str(block.value)
        new_value = current_value + "\n" + str(content)
        
        if len(new_value) > block.limit:
            raise ValueError(
                f"Edit failed: Exceeds {block.limit} character limit "
                f"(current: {len(current_value)}, adding: {len(content)}, "
                f"total: {len(new_value)})"
            )
        
        self.memory.update_block_value(label, new_value)
        return None
    
    def core_memory_replace(
        self, 
        label: str, 
        old_content: str, 
        new_content: str
    ) -> Optional[str]:
        """
        Replace specific text in a memory block.
        To delete text, use an empty string for new_content.
        
        Args:
            label: The memory block to edit
            old_content: Exact text to find and replace (must match exactly)
            new_content: Text to replace with (can be empty string to delete)
            
        Returns:
            None (no message returned to LLM)
            
        Example:
            core_memory_replace(
                label="human",
                old_content="Alice prefers light mode",
                new_content="Alice prefers dark mode"
            )
        """
        block = self.memory.get_block(label)
        
        if block.read_only:
            raise ValueError(f"Block '{label}' is read-only and cannot be edited")
        
        current_value = str(block.value)
        
        if old_content not in current_value:
            raise ValueError(
                f"Old content '{old_content}' not found in memory block '{label}'"
            )
        
        new_value = current_value.replace(str(old_content), str(new_content))
        
        if len(new_value) > block.limit:
            raise ValueError(
                f"Edit failed: Exceeds {block.limit} character limit "
                f"(result would be {len(new_value)} characters)"
            )
        
        self.memory.update_block_value(label, new_value)
        return None
    
    # ===== ADVANCED MEMORY TOOLS (v2) =====
    
    def memory_replace(
        self, 
        label: str, 
        old_str: str, 
        new_str: str
    ) -> str:
        """
        Replace a specific string in a memory block with validation.
        More robust than core_memory_replace with better error messages.
        
        Args:
            label: Block label to edit
            old_str: Text to replace (must be unique in the block)
            new_str: Text to replace with
            
        Returns:
            Success message with editing guidance
            
        Raises:
            ValueError: If old_str not found, not unique, or contains line numbers
        """
        block = self.memory.get_block(label)
        
        if block.read_only:
            raise ValueError(f"Block '{label}' is read-only")
        
        # Validate no line numbers in arguments (common LLM error)
        if LINE_NUMBER_REGEX.search(old_str):
            raise ValueError(
                "old_str contains line number prefix (e.g., '1→'). "
                "Do not include line numbers - they are for display only."
            )
        
        if LINE_NUMBER_REGEX.search(new_str):
            raise ValueError(
                "new_str contains line number prefix (e.g., '1→'). "
                "Do not include line numbers - they are for display only."
            )
        
        # Expand tabs for consistent matching
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs()
        current_value = block.value.expandtabs()
        
        # Check if old_str exists and is unique
        occurrences = current_value.count(old_str)
        
        if occurrences == 0:
            raise ValueError(
                f"Text '{old_str[:50]}{'...' if len(old_str) > 50 else ''}' "
                f"not found in memory block '{label}'"
            )
        elif occurrences > 1:
            # Find line numbers where it appears
            lines = [
                idx + 1 
                for idx, line in enumerate(current_value.split("\n"))
                if old_str in line
            ]
            raise ValueError(
                f"Text appears {occurrences} times in block '{label}' "
                f"(on lines: {lines}). Include more surrounding context "
                f"to make the match unique."
            )
        
        # Perform replacement
        new_value = current_value.replace(old_str, new_str)
        
        if len(new_value) > block.limit:
            raise ValueError(
                f"Edit failed: Result ({len(new_value)} chars) exceeds "
                f"{block.limit} character limit"
            )
        
        self.memory.update_block_value(label, new_value)
        
        return (
            f"Successfully edited memory block '{label}'. "
            f"Review the changes and make additional edits if needed."
        )
    
    def memory_insert(
        self, 
        label: str, 
        new_str: str, 
        insert_line: int = -1
    ) -> str:
        """
        Insert text at a specific line in a memory block.
        
        Args:
            label: Block label to edit
            new_str: Text to insert
            insert_line: Line number to insert after (0 = beginning, -1 = end)
            
        Returns:
            Success message
            
        Example:
            # Insert at beginning
            memory_insert(label="human", new_str="Priority: Bug fixes", insert_line=0)
            
            # Append to end
            memory_insert(label="human", new_str="New preference noted", insert_line=-1)
        """
        block = self.memory.get_block(label)
        
        if block.read_only:
            raise ValueError(f"Block '{label}' is read-only")
        
        # Validate no line numbers
        if LINE_NUMBER_REGEX.search(new_str):
            raise ValueError(
                "new_str contains line number prefix. "
                "Do not include line numbers - they are for display only."
            )
        
        current_value = block.value.expandtabs()
        new_str = new_str.expandtabs()
        current_lines = current_value.split("\n")
        n_lines = len(current_lines)
        
        # Handle -1 as append to end
        if insert_line == -1:
            insert_line = n_lines
        elif insert_line < 0 or insert_line > n_lines:
            raise ValueError(
                f"Invalid insert_line: {insert_line}. "
                f"Must be between 0 and {n_lines}, or -1 for end."
            )
        
        # Insert the new lines
        new_str_lines = new_str.split("\n")
        new_lines = (
            current_lines[:insert_line] + 
            new_str_lines + 
            current_lines[insert_line:]
        )
        new_value = "\n".join(new_lines)
        
        if len(new_value) > block.limit:
            raise ValueError(
                f"Edit failed: Result ({len(new_value)} chars) exceeds "
                f"{block.limit} character limit"
            )
        
        self.memory.update_block_value(label, new_value)
        
        return (
            f"Successfully inserted text into memory block '{label}' "
            f"at line {insert_line}."
        )
    
    def memory_rethink(self, label: str, new_memory: str) -> str:
        """
        Completely rewrite a memory block from scratch.
        Use this for major reorganization, not small edits.
        
        Args:
            label: Block label to rewrite
            new_memory: Complete new contents for the block
            
        Returns:
            Success message
            
        Example:
            memory_rethink(
                label="human",
                new_memory=\"\"\"User: Alice
                Role: Software Engineer
                Focus: Learning AI agents
                Preferences:
                - Python examples
                - Detailed comments
                \"\"\"
            )
        """
        # Validate no line numbers
        if LINE_NUMBER_REGEX.search(new_memory):
            raise ValueError(
                "new_memory contains line number prefix. "
                "Do not include line numbers - they are for display only."
            )
        
        try:
            block = self.memory.get_block(label)
            if block.read_only:
                raise ValueError(f"Block '{label}' is read-only")
        except KeyError:
            # Block doesn't exist - create it
            new_block = Block(label=label, value="", limit=2000)
            self.memory.set_block(new_block)
            block = self.memory.get_block(label)
        
        if len(new_memory) > block.limit:
            raise ValueError(
                f"Edit failed: New content ({len(new_memory)} chars) exceeds "
                f"{block.limit} character limit"
            )
        
        self.memory.update_block_value(label, new_memory)
        
        return (
            f"Successfully rewrote memory block '{label}'. "
            f"Review the changes carefully."
        )
    
    # ===== ARCHIVAL MEMORY TOOLS =====
    
    def archival_memory_insert(
        self, 
        content: str, 
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Store information in long-term archival memory.
        This is separate from core memory and searchable via embeddings.
        
        Args:
            content: Information to store (self-contained fact or summary)
            tags: Optional category tags for organization
            
        Returns:
            Confirmation message with passage ID
        """
        if not self.archival_manager:
            raise NotImplementedError(
                "Archival memory not configured. "
                "Provide an ArchivalMemoryManager instance."
            )
        
        passage_id = self.archival_manager.insert(
            agent_id=self.agent_id,
            content=content,
            tags=tags
        )
        
        return f"Successfully stored in archival memory (ID: {passage_id})"
    
    def archival_memory_search(
        self, 
        query: str, 
        top_k: int = 5
    ) -> str:
        """
        Search archival memory using semantic similarity.
        
        Args:
            query: What to search for (natural language)
            top_k: Maximum results to return (default: 5)
            
        Returns:
            Formatted string with search results
        """
        if not self.archival_manager:
            raise NotImplementedError("Archival memory not configured")
        
        results = self.archival_manager.search(
            agent_id=self.agent_id,
            query=query,
            top_k=top_k
        )
        
        if not results:
            return "No results found in archival memory."
        
        output = f"Found {len(results)} results:\n\n"
        for i, result in enumerate(results, 1):
            output += f"[{i}] {result['content']}\n"
            if result.get('timestamp'):
                output += f"    Stored: {result['timestamp']}\n"
            if result.get('tags'):
                output += f"    Tags: {', '.join(result['tags'])}\n"
            output += "\n"
        
        return output
    
    # ===== CONVERSATION SEARCH TOOLS =====
    
    def conversation_search(
        self,
        query: str,
        page: int = 0,
        limit: int = 10
    ) -> str:
        """
        Search past conversation history using text matching.
        
        Args:
            query: What to search for
            page: Page number for pagination (0-indexed)
            limit: Results per page
            
        Returns:
            Formatted string with search results and timestamps
        """
        if not self.message_manager:
            raise NotImplementedError("Message manager not configured")
        
        results = self.message_manager.search_messages(
            agent_id=self.agent_id,
            query=query,
            limit=limit,
            offset=page * limit
        )
        
        if not results:
            return f"No messages found matching '{query}'."
        
        output = f"Found messages matching '{query}':\n\n"
        for msg in results:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("created_at", "")
            
            # Truncate long messages
            if len(content) > 200:
                content = content[:200] + "..."
            
            output += f"[{timestamp}] {role}: {content}\n\n"
        
        return output


# ===== TOOL SCHEMAS FOR LLM =====

def get_memory_tool_schemas(include_heartbeat: bool = True) -> List[dict]:
    """
    Return OpenAI-format function schemas for memory tools.
    These are passed to the LLM to enable function calling.
    
    Args:
        include_heartbeat: Whether to include request_heartbeat parameter
        
    Returns:
        List of tool schemas in OpenAI format
    """
    
    def add_heartbeat(params: dict) -> dict:
        """Add request_heartbeat parameter to tool schema"""
        if include_heartbeat:
            params["properties"]["request_heartbeat"] = {
                "type": "boolean",
                "description": (
                    "Request an immediate follow-up response. "
                    "Set to true if you want to continue processing."
                ),
            }
        return params
    
    return [
        {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "Send a message to the user",
                "parameters": add_heartbeat({
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to send to user",
                        },
                    },
                    "required": ["message"],
                }),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "core_memory_append",
                "description": "Append content to a core memory block",
                "parameters": add_heartbeat({
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Block label (e.g., 'human', 'persona')",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to append",
                        },
                    },
                    "required": ["label", "content"],
                }),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "core_memory_replace",
                "description": "Replace specific text in a core memory block",
                "parameters": add_heartbeat({
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Block label to edit",
                        },
                        "old_content": {
                            "type": "string",
                            "description": "Exact text to find and replace",
                        },
                        "new_content": {
                            "type": "string",
                            "description": "Replacement text (empty string to delete)",
                        },
                    },
                    "required": ["label", "old_content", "new_content"],
                }),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_rethink",
                "description": "Completely rewrite a memory block (for major reorganization only)",
                "parameters": add_heartbeat({
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Block label to rewrite",
                        },
                        "new_memory": {
                            "type": "string",
                            "description": "Complete new contents for the block",
                        },
                    },
                    "required": ["label", "new_memory"],
                }),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "archival_memory_insert",
                "description": "Store information in long-term archival memory",
                "parameters": add_heartbeat({
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Information to store",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for categorization",
                        },
                    },
                    "required": ["content"],
                }),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "archival_memory_search",
                "description": "Search archival memory using semantic similarity",
                "parameters": add_heartbeat({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum results to return",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                }),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "conversation_search",
                "description": "Search past conversation history",
                "parameters": add_heartbeat({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "page": {
                            "type": "integer",
                            "description": "Page number (0-indexed)",
                            "default": 0,
                        },
                    },
                    "required": ["query"],
                }),
            },
        },
    ]
```

---

## Message Management

### MessageManager - Production Message Handling

```python
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
import uuid
import json

class MessageManager:
    """
    Manages message persistence and retrieval.
    
    This is how Letta actually handles messages at scale.
    Messages are stored in the database, and the Agent only holds
    message IDs in its state.
    """
    
    def __init__(self, db: 'MemoryDatabase'):
        self.db = db
    
    def create_message(
        self,
        agent_id: str,
        role: str,
        content: Union[str, List[Dict]],
        tool_calls: Optional[List[Dict]] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Create and persist a message.
        
        Args:
            agent_id: Agent this message belongs to
            role: Message role (user, assistant, tool, system)
            content: Message content (string or list of content objects)
            tool_calls: Tool calls made by assistant
            tool_call_id: ID of tool call this message responds to
            name: Name of function (for tool messages)
            model: Model that generated this message
            
        Returns:
            message_id: Unique identifier for the message
        """
        message_id = f"message-{uuid.uuid4()}"
        
        # Normalize content to list of content objects
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        
        with self.db._get_connection() as conn:
            conn.execute("""
                INSERT INTO messages (
                    id, agent_id, role, content, tool_calls, 
                    tool_call_id, name, model, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message_id,
                agent_id,
                role,
                json.dumps(content),
                json.dumps(tool_calls) if tool_calls else None,
                tool_call_id,
                name,
                model,
                datetime.utcnow().isoformat()
            ))
            conn.commit()
        
        return message_id
    
    def get_messages_by_ids(
        self,
        message_ids: List[str],
        agent_id: str
    ) -> List[Dict]:
        """
        Get messages by IDs in the order specified.
        
        This is how you load in-context messages for an agent step.
        The order of message_ids is preserved.
        
        Args:
            message_ids: List of message IDs to retrieve
            agent_id: Agent ID for validation
            
        Returns:
            List of message dictionaries in the specified order
        """
        if not message_ids:
            return []
        
        with self.db._get_connection() as conn:
            placeholders = ','.join('?' * len(message_ids))
            
            cursor = conn.execute(f"""
                SELECT *
                FROM messages
                WHERE id IN ({placeholders})
                AND agent_id = ?
            """, (*message_ids, agent_id))
            
            # Create a map for O(1) lookup
            message_map = {}
            for row in cursor.fetchall():
                msg = {
                    "id": row["id"],
                    "role": row["role"],
                    "content": json.loads(row["content"]),
                    "created_at": row["created_at"],
                }
                
                if row["tool_calls"]:
                    msg["tool_calls"] = json.loads(row["tool_calls"])
                if row["tool_call_id"]:
                    msg["tool_call_id"] = row["tool_call_id"]
                if row["name"]:
                    msg["name"] = row["name"]
                if row["model"]:
                    msg["model"] = row["model"]
                
                message_map[row["id"]] = msg
            
            # Return in the order specified by message_ids
            return [message_map[mid] for mid in message_ids if mid in message_map]
    
    def append_to_in_context(
        self,
        agent_state: AgentState,
        messages: List[Dict]
    ) -> AgentState:
        """
        Add messages to agent's in-context window and update state.
        
        This is the key method called after every agent step.
        
        Args:
            agent_state: Current agent state
            messages: New messages to add
            
        Returns:
            Updated agent state with new message IDs
        """
        new_message_ids = []
        
        for msg in messages:
            message_id = self.create_message(
                agent_id=agent_state.id,
                role=msg["role"],
                content=msg.get("content", ""),
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id"),
                name=msg.get("name"),
                model=msg.get("model")
            )
            new_message_ids.append(message_id)
        
        # Update agent state
        agent_state.message_ids.extend(new_message_ids)
        agent_state.updated_at = datetime.utcnow()
        
        return agent_state
    
    def trim_messages(
        self,
        agent_state: AgentState,
        num_to_remove: int
    ) -> AgentState:
        """
        Remove oldest messages from in-context window.
        Used during summarization to free up context space.
        
        Note: Always keeps the system message (index 0).
        
        Args:
            agent_state: Current agent state
            num_to_remove: Number of messages to remove
            
        Returns:
            Updated agent state with trimmed message IDs
        """
        if num_to_remove >= len(agent_state.message_ids) - 1:
            raise ValueError(
                f"Cannot remove {num_to_remove} messages - "
                f"would remove system message or all messages"
            )
        
        # Keep system message (index 0), remove from index 1
        agent_state.message_ids = (
            [agent_state.message_ids[0]] +  # Keep system
            agent_state.message_ids[num_to_remove + 1:]  # Skip removed
        )
        agent_state.updated_at = datetime.utcnow()
        
        return agent_state
    
    def prepend_to_in_context(
        self,
        agent_state: AgentState,
        messages: List[Dict]
    ) -> AgentState:
        """
        Prepend messages after system message (for summaries).
        
        Used when inserting a summary message at the beginning
        of the conversation (after the system message).
        """
        new_message_ids = []
        
        for msg in messages:
            message_id = self.create_message(
                agent_id=agent_state.id,
                role=msg["role"],
                content=msg.get("content", ""),
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id"),
                name=msg.get("name")
            )
            new_message_ids.append(message_id)
        
        # Insert after system message
        agent_state.message_ids = (
            [agent_state.message_ids[0]] +  # System message
            new_message_ids +                # New messages (summary)
            agent_state.message_ids[1:]      # Rest of messages
        )
        agent_state.updated_at = datetime.utcnow()
        
        return agent_state
    
    def search_messages(
        self,
        agent_id: str,
        query: str,
        limit: int = 10,
        offset: int = 0,
        roles: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search messages by content.
        
        For production, consider using full-text search (FTS5)
        or an external search engine.
        """
        with self.db._get_connection() as conn:
            role_filter = ""
            params = [agent_id, f"%{query}%"]
            
            if roles:
                role_placeholders = ','.join('?' * len(roles))
                role_filter = f"AND role IN ({role_placeholders})"
                params.extend(roles)
            
            params.extend([limit, offset])
            
            cursor = conn.execute(f"""
                SELECT id, role, content, created_at
                FROM messages
                WHERE agent_id = ?
                AND content LIKE ?
                {role_filter}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, params)
            
            return [
                {
                    "id": row["id"],
                    "role": row["role"],
                    "content": json.loads(row["content"]),
                    "created_at": row["created_at"],
                }
                for row in cursor.fetchall()
            ]
    
    def size(self, agent_id: str) -> int:
        """Get total number of messages for an agent"""
        with self.db._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE agent_id = ?",
                (agent_id,)
            )
            return cursor.fetchone()[0]
```

---

## Context Window Management

### ContextWindowManager - Automatic Summarization

This is **critical** for production and often overlooked:

```python
from typing import List, Dict, Optional, Tuple
import tiktoken

class ContextWindowManager:
    """
    Manages context window overflow and automatic summarization.
    
    This is Letta's killer feature for handling infinite conversations.
    When the context window fills up, older messages are automatically
    summarized and replaced with a summary message.
    """
    
    def __init__(
        self,
        message_manager: MessageManager,
        llm_client: Any,
        context_window_limit: int = 8000,  # tokens
        memory_warning_threshold: float = 0.75,
        desired_memory_pressure: float = 0.3
    ):
        self.message_manager = message_manager
        self.llm_client = llm_client
        self.context_window_limit = context_window_limit
        self.memory_warning_threshold = memory_warning_threshold
        self.desired_memory_pressure = desired_memory_pressure
        
        # Token encoder
        try:
            self._encoding = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, messages: List[Dict]) -> int:
        """
        Count tokens in a list of messages.
        
        Uses tiktoken for accurate counts.
        """
        total = 0
        
        for msg in messages:
            # Role overhead
            total += 4  # Every message has ~4 tokens of overhead
            
            # Content
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(self._encoding.encode(content))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += len(self._encoding.encode(item.get("text", "")))
            
            # Tool calls
            if msg.get("tool_calls"):
                total += len(self._encoding.encode(json.dumps(msg["tool_calls"])))
            
            # Name
            if msg.get("name"):
                total += len(self._encoding.encode(msg["name"]))
        
        return total
    
    def check_context_overflow(
        self,
        agent_state: AgentState
    ) -> Tuple[bool, float]:
        """
        Check if we're approaching context window limit.
        
        Args:
            agent_state: Current agent state
            
        Returns:
            (is_overflow, current_usage_percentage)
        """
        messages = self.message_manager.get_messages_by_ids(
            agent_state.message_ids,
            agent_state.id
        )
        
        token_count = self.count_tokens(messages)
        usage = token_count / self.context_window_limit
        
        is_overflow = usage > self.memory_warning_threshold
        
        if is_overflow:
            print(f"⚠️  Context window {usage:.0%} full "
                  f"({token_count}/{self.context_window_limit} tokens)")
        
        return is_overflow, usage
    
    def calculate_summarize_cutoff(
        self,
        messages: List[Dict],
        token_counts: List[int]
    ) -> int:
        """
        Calculate how many messages to summarize.
        
        Aims to reduce context to desired_memory_pressure of limit.
        """
        total_tokens = sum(token_counts)
        target_tokens = int(self.context_window_limit * self.desired_memory_pressure)
        tokens_to_remove = total_tokens - target_tokens
        
        if tokens_to_remove <= 0:
            return 1  # Summarize at least 1 message
        
        # Find cutoff point
        cumulative = 0
        for i, count in enumerate(token_counts[1:], start=1):  # Skip system
            cumulative += count
            if cumulative >= tokens_to_remove:
                return i
        
        # Don't summarize everything - keep at least a few recent messages
        return max(1, len(messages) - 5)
    
    def summarize_messages_inplace(
        self,
        agent_state: AgentState,
        num_to_summarize: Optional[int] = None
    ) -> AgentState:
        """
        Summarize older messages to free up context space.
        
        This is Letta's approach to handling infinite conversations.
        
        Args:
            agent_state: Current agent state
            num_to_summarize: Number of messages to summarize (auto-calculated if None)
            
        Returns:
            Updated agent state with summarized messages
        """
        # Get all in-context messages
        messages = self.message_manager.get_messages_by_ids(
            agent_state.message_ids,
            agent_state.id
        )
        
        if len(messages) <= 2:  # System + at most 1 message
            raise ValueError("Not enough messages to summarize")
        
        # Calculate token counts
        token_counts = [self.count_tokens([m]) for m in messages]
        
        # Determine cutoff
        if num_to_summarize is None:
            cutoff = self.calculate_summarize_cutoff(messages, token_counts)
        else:
            cutoff = num_to_summarize
        
        # Messages to summarize (skip system message at index 0)
        messages_to_summarize = messages[1:cutoff + 1]
        
        if not messages_to_summarize:
            raise ValueError("No messages to summarize")
        
        print(f"📝 Summarizing {len(messages_to_summarize)} of {len(messages)} messages...")
        
        # Generate summary
        summary = self._generate_summary(messages_to_summarize)
        
        # Create summary message
        total_messages = self.message_manager.size(agent_state.id)
        remaining_messages = len(messages) - cutoff
        hidden_messages = total_messages - remaining_messages
        
        summary_content = (
            f"[Summary of the previous {len(messages_to_summarize)} messages "
            f"({hidden_messages} total messages hidden)]\n\n"
            f"{summary}\n\n"
            f"[End of summary. The following are the {remaining_messages - 1} "
            f"most recent messages.]"
        )
        
        # Trim old messages and prepend summary
        agent_state = self.message_manager.trim_messages(agent_state, cutoff)
        agent_state = self.message_manager.prepend_to_in_context(
            agent_state,
            [{"role": "user", "content": summary_content}]
        )
        
        # Log results
        new_messages = self.message_manager.get_messages_by_ids(
            agent_state.message_ids,
            agent_state.id
        )
        new_token_count = self.count_tokens(new_messages)
        old_token_count = sum(token_counts)
        
        print(f"✅ Summarization complete: {len(messages)} → {len(new_messages)} messages")
        print(f"   Tokens: {old_token_count} → {new_token_count} "
              f"({(1 - new_token_count/old_token_count)*100:.0f}% reduction)")
        
        return agent_state
    
    def _generate_summary(self, messages: List[Dict]) -> str:
        """Generate a summary of messages using LLM"""
        
        # Format messages for summarization
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "") 
                    for item in content 
                    if isinstance(item, dict)
                )
            
            formatted.append(f"[{role}]: {content}")
        
        messages_text = "\n\n".join(formatted)
        
        prompt = f"""Summarize the following conversation segment concisely.
Preserve:
- Key facts and decisions
- User preferences and requirements  
- Important context for future conversation
- Any commitments or action items

Conversation:
{messages_text}

Summary (2-3 paragraphs):"""
        
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",  # Use cheaper model for summarization
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
```

---

## Archival Memory (Vector Search)

### ArchivalMemoryManager - Complete Implementation

```python
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid
import json

class ArchivalMemoryManager:
    """
    Manages long-term archival memory with vector embeddings.
    
    This provides unlimited storage for information that doesn't fit
    in core memory, searchable via semantic similarity.
    
    Can be integrated with:
    - Pinecone
    - Weaviate
    - ChromaDB
    - pgvector
    - Qdrant
    """
    
    def __init__(
        self,
        db: 'MemoryDatabase',
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = 1536,
        llm_client: Any = None,
        vector_store: Any = None  # Optional external vector DB
    ):
        self.db = db
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.llm_client = llm_client
        self.vector_store = vector_store
    
    def insert(
        self,
        agent_id: str,
        content: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Insert text into archival memory with embedding.
        
        Args:
            agent_id: Agent this memory belongs to
            content: Text content to store
            tags: Optional categorization tags
            metadata: Optional additional metadata
            
        Returns:
            passage_id: Unique identifier for the passage
        """
        passage_id = f"passage-{uuid.uuid4()}"
        
        # Generate embedding if we have an LLM client
        embedding = None
        if self.llm_client:
            embedding = self._generate_embedding(content)
        
        # Save to local database
        with self.db._get_connection() as conn:
            conn.execute("""
                INSERT INTO archival_memory (
                    id, agent_id, content, tags, metadata, 
                    embedding, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                passage_id,
                agent_id,
                content,
                json.dumps(tags) if tags else None,
                json.dumps(metadata) if metadata else None,
                json.dumps(embedding) if embedding else None,
                datetime.utcnow().isoformat()
            ))
            conn.commit()
        
        # Save to external vector store if configured
        if self.vector_store and embedding:
            self.vector_store.upsert(
                vectors=[{
                    "id": passage_id,
                    "values": embedding,
                    "metadata": {
                        "agent_id": agent_id,
                        "content": content[:1000],  # Truncate for metadata
                        "tags": tags or [],
                        "created_at": datetime.utcnow().isoformat(),
                    }
                }]
            )
        
        return passage_id
    
    def search(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5,
        tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search archival memory using semantic similarity.
        
        Args:
            agent_id: Agent to search within
            query: Search query (natural language)
            top_k: Maximum results to return
            tags: Optional tag filter
            
        Returns:
            List of matching passages with scores
        """
        # Try vector search first
        if self.vector_store and self.llm_client:
            return self._vector_search(agent_id, query, top_k, tags)
        
        # Fallback to text search
        return self._text_search(agent_id, query, top_k, tags)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        response = self.llm_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def _vector_search(
        self,
        agent_id: str,
        query: str,
        top_k: int,
        tags: Optional[List[str]]
    ) -> List[Dict]:
        """Search using vector similarity"""
        query_embedding = self._generate_embedding(query)
        
        # Build filter
        filter_dict = {"agent_id": agent_id}
        if tags:
            filter_dict["tags"] = {"$in": tags}
        
        results = self.vector_store.query(
            vector=query_embedding,
            filter=filter_dict,
            top_k=top_k,
            include_metadata=True
        )
        
        return [
            {
                "id": match["id"],
                "content": match["metadata"].get("content", ""),
                "score": match["score"],
                "tags": match["metadata"].get("tags", []),
                "timestamp": match["metadata"].get("created_at"),
            }
            for match in results.get("matches", [])
        ]
    
    def _text_search(
        self,
        agent_id: str,
        query: str,
        top_k: int,
        tags: Optional[List[str]]
    ) -> List[Dict]:
        """Fallback text search using SQLite"""
        with self.db._get_connection() as conn:
            tag_filter = ""
            params = [agent_id, f"%{query}%"]
            
            if tags:
                # Simple JSON containment check
                tag_conditions = " OR ".join(
                    "tags LIKE ?" for _ in tags
                )
                tag_filter = f"AND ({tag_conditions})"
                params.extend([f'%"{tag}"%' for tag in tags])
            
            params.append(top_k)
            
            cursor = conn.execute(f"""
                SELECT id, content, tags, created_at
                FROM archival_memory
                WHERE agent_id = ?
                AND content LIKE ?
                {tag_filter}
                ORDER BY created_at DESC
                LIMIT ?
            """, params)
            
            return [
                {
                    "id": row["id"],
                    "content": row["content"],
                    "tags": json.loads(row["tags"]) if row["tags"] else [],
                    "timestamp": row["created_at"],
                    "score": 0.0,  # No score for text search
                }
                for row in cursor.fetchall()
            ]
    
    def delete(self, passage_id: str, agent_id: str) -> bool:
        """Delete a passage from archival memory"""
        with self.db._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM archival_memory WHERE id = ? AND agent_id = ?",
                (passage_id, agent_id)
            )
            conn.commit()
            deleted = cursor.rowcount > 0
        
        # Also delete from vector store
        if self.vector_store and deleted:
            self.vector_store.delete(ids=[passage_id])
        
        return deleted
    
    def count(self, agent_id: str) -> int:
        """Count passages for an agent"""
        with self.db._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM archival_memory WHERE agent_id = ?",
                (agent_id,)
            )
            return cursor.fetchone()[0]
```

---

## Agent Integration

### Building the System Prompt

```python
class PromptGenerator:
    """
    Builds the complete system prompt including memory, metadata, and instructions.
    
    This is regenerated on every agent step to include the latest memory state.
    """
    
    @staticmethod
    def build_system_message(
        agent_state: AgentState,
        memory: Memory,
        message_manager: MessageManager,
        archival_manager: Optional[ArchivalMemoryManager] = None
    ) -> str:
        """
        Build complete system message including all context.
        
        Args:
            agent_state: Current agent state
            memory: Memory object with blocks
            message_manager: For message statistics
            archival_manager: For archival statistics
            
        Returns:
            Complete system prompt string
        """
        parts = []
        
        # 1. Base system instructions
        parts.append(agent_state.system)
        parts.append("")
        
        # 2. Memory metadata (critical for agent awareness!)
        parts.append("### Memory Statistics")
        parts.append(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %I:%M:%S %p UTC')}")
        
        total_messages = message_manager.size(agent_state.id)
        parts.append(f"Total messages in conversation: {total_messages}")
        
        if archival_manager:
            archival_count = archival_manager.count(agent_state.id)
            parts.append(f"Archival memory entries: {archival_count}")
        
        parts.append("")
        
        # 3. Core memory blocks
        memory_str = memory.compile(
            include_line_numbers=agent_state.agent_type == "sleeptime_agent",
            llm_config=agent_state.llm_config
        )
        parts.append(memory_str)
        
        return "\n".join(parts)
    
    @staticmethod
    def get_default_system_prompt() -> str:
        """Get default base system instructions"""
        return """You are a helpful AI assistant with advanced memory capabilities.

<memory_system>
You have a sophisticated memory system with three tiers:

1. **Core Memory** (always visible):
   - Memory blocks shown below containing key information
   - You can edit these using memory tools
   - Has character limits - be concise

2. **Recall Memory** (searchable):
   - Full conversation history
   - Search using conversation_search tool
   
3. **Archival Memory** (long-term):
   - Unlimited storage for important information
   - Semantic search using archival_memory_search
   - Store important facts using archival_memory_insert
</memory_system>

<memory_guidelines>
Update your memory when you learn:
- New information about the user (name, preferences, background)
- Important facts or decisions from the conversation
- User corrections to previous information
- Context that will be useful in future conversations

Keep memory:
- Organized and structured
- Concise (respect character limits)
- Up-to-date (remove outdated information)
- Focused on persistent facts, not temporary context
</memory_guidelines>

<tool_guidelines>
- Always call send_message to respond to the user
- Use core_memory_replace for small, precise edits
- Use memory_rethink only for major reorganization
- Search archival/recall memory before asking the user for info they may have shared
</tool_guidelines>
"""
```

### The Complete Agent

```python
from typing import List, Dict, Any, Optional
import json
import traceback

class Agent:
    """
    Complete Letta-style agent with proper state management.
    
    Separates stateful Agent from serializable AgentState.
    Handles tool calling, memory management, and context window overflow.
    """
    
    def __init__(
        self,
        agent_state: AgentState,
        memory: Memory,
        llm_client: Any,
        message_manager: MessageManager,
        context_manager: ContextWindowManager,
        archival_manager: Optional[ArchivalMemoryManager] = None,
        db: Optional['MemoryDatabase'] = None
    ):
        self.agent_state = agent_state
        self.memory = memory
        self.llm_client = llm_client
        self.message_manager = message_manager
        self.context_manager = context_manager
        self.archival_manager = archival_manager
        self.db = db
        
        # Attach memory to agent state
        self.agent_state.memory = memory
        
        # Initialize memory tools
        self.memory_tools = MemoryTools(
            memory=memory,
            archival_manager=archival_manager,
            message_manager=message_manager,
            agent_id=agent_state.id
        )
        
        # Tool registry
        self.tools = {
            "send_message": self._send_message,
            "core_memory_append": self.memory_tools.core_memory_append,
            "core_memory_replace": self.memory_tools.core_memory_replace,
            "memory_replace": self.memory_tools.memory_replace,
            "memory_insert": self.memory_tools.memory_insert,
            "memory_rethink": self.memory_tools.memory_rethink,
            "archival_memory_insert": self.memory_tools.archival_memory_insert,
            "archival_memory_search": self.memory_tools.archival_memory_search,
            "conversation_search": self.memory_tools.conversation_search,
        }
    
    def _send_message(self, message: str) -> str:
        """Tool for sending a message to the user"""
        return message
    
    def step(
        self,
        user_message: str,
        max_chaining_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Execute agent step with proper Letta semantics.
        
        Args:
            user_message: Message from the user
            max_chaining_steps: Maximum tool calling iterations
            
        Returns:
            {
                "messages": List of new messages,
                "response": Final response to user,
                "usage": Token usage stats,
                "step_count": Number of LLM calls made
            }
        """
        # 1. Check context window BEFORE adding user message
        is_overflow, usage = self.context_manager.check_context_overflow(self.agent_state)
        if is_overflow:
            print("🔄 Triggering automatic summarization...")
            self.agent_state = self.context_manager.summarize_messages_inplace(
                self.agent_state
            )
            self._save_state()
        
        # 2. Create and persist user message
        user_msg = {"role": "user", "content": user_message}
        self.agent_state = self.message_manager.append_to_in_context(
            self.agent_state,
            [user_msg]
        )
        
        # 3. Agent loop (with tool chaining)
        step_count = 0
        all_new_messages = [user_msg]
        total_tokens = 0
        final_response = None
        
        while step_count < max_chaining_steps:
            step_count += 1
            
            # 4. Rebuild system prompt (includes latest memory)
            system_prompt = PromptGenerator.build_system_message(
                agent_state=self.agent_state,
                memory=self.memory,
                message_manager=self.message_manager,
                archival_manager=self.archival_manager
            )
            
            # 5. Get in-context messages
            in_context_messages = self.message_manager.get_messages_by_ids(
                self.agent_state.message_ids,
                self.agent_state.id
            )
            
            # 6. Prepare LLM messages
            llm_messages = [
                {"role": "system", "content": system_prompt},
                *[self._format_for_llm(m) for m in in_context_messages[1:]]
            ]
            
            # 7. Call LLM
            response = self._call_llm(llm_messages)
            
            if response.get("usage"):
                total_tokens += response["usage"].get("total_tokens", 0)
            
            # 8. Process response
            assistant_msg = {
                "role": "assistant",
                "content": response.get("content", "")
            }
            
            if response.get("tool_calls"):
                assistant_msg["tool_calls"] = response["tool_calls"]
            
            all_new_messages.append(assistant_msg)
            
            # 9. Persist assistant message
            self.agent_state = self.message_manager.append_to_in_context(
                self.agent_state,
                [assistant_msg]
            )
            
            # 10. If no tool calls, we're done
            if not response.get("tool_calls"):
                final_response = response.get("content", "")
                break
            
            # 11. Execute tools
            heartbeat_request = False
            
            for tool_call in response["tool_calls"]:
                tool_result_msg, is_heartbeat, result = self._execute_tool(tool_call)
                all_new_messages.append(tool_result_msg)
                
                # Check for send_message result
                if tool_call["function"]["name"] == "send_message" and result:
                    final_response = result
                
                # Persist tool result
                self.agent_state = self.message_manager.append_to_in_context(
                    self.agent_state,
                    [tool_result_msg]
                )
                
                if is_heartbeat:
                    heartbeat_request = True
            
            # 12. Continue if heartbeat requested, otherwise stop
            if not heartbeat_request:
                break
        
        # 13. Save final state
        self._save_state()
        
        # 14. Return results
        return {
            "messages": all_new_messages,
            "response": final_response,
            "usage": {"total_tokens": total_tokens},
            "step_count": step_count
        }
    
    def _execute_tool(self, tool_call: Dict) -> tuple:
        """
        Execute a tool and return result message.
        
        Returns:
            (tool_result_message, heartbeat_requested, result_value)
        """
        tool_name = tool_call["function"]["name"]
        
        try:
            # Parse arguments
            args = json.loads(tool_call["function"]["arguments"])
            
            # Extract heartbeat request
            heartbeat = args.pop("request_heartbeat", False)
            
            # Execute tool
            if tool_name in self.tools:
                result = self.tools[tool_name](**args)
                result_str = str(result) if result else "Success"
            else:
                result_str = f"Error: Unknown tool '{tool_name}'"
                result = None
            
            print(f"[TOOL] {tool_name}({json.dumps(args)[:100]}...)")
            print(f"[RESULT] {result_str[:100]}...")
            
        except Exception as e:
            result_str = f"Error: {str(e)}\n{traceback.format_exc()}"
            result = None
            heartbeat = True  # Force retry on error
            print(f"[TOOL ERROR] {tool_name}: {e}")
        
        # Create tool result message
        tool_msg = {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "name": tool_name,
            "content": result_str
        }
        
        return tool_msg, heartbeat, result
    
    def _call_llm(self, messages: List[Dict]) -> Dict:
        """Call LLM with tools enabled"""
        response = self.llm_client.chat.completions.create(
            model=self.agent_state.llm_config.model,
            messages=messages,
            tools=get_memory_tool_schemas(),
            tool_choice="auto",
            temperature=self.agent_state.llm_config.temperature
        )
        
        message = response.choices[0].message
        
        result = {
            "content": message.content,
            "tool_calls": [],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }
        
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        
        return result
    
    def _format_for_llm(self, msg: Dict) -> Dict:
        """Convert internal message format to LLM API format"""
        content = msg.get("content", "")
        
        # Convert content list to string if needed
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") 
                for item in content 
                if isinstance(item, dict) and item.get("type") == "text"
            )
        
        formatted = {"role": msg["role"], "content": content}
        
        if msg.get("tool_calls"):
            formatted["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id"):
            formatted["tool_call_id"] = msg["tool_call_id"]
        if msg.get("name"):
            formatted["name"] = msg["name"]
        
        return formatted
    
    def _save_state(self):
        """Persist agent state and memory to database"""
        if not self.db:
            return
        
        # Save all memory blocks
        for block in self.memory.blocks:
            self.db.save_block(block)
        
        # Save agent state
        self.db.save_agent_state(self.agent_state)
    
    def get_memory_state(self) -> Dict:
        """Get current memory state for inspection"""
        return {
            "blocks": [
                {
                    "label": b.label,
                    "value": b.value,
                    "chars": len(b),
                    "limit": b.limit,
                    "read_only": b.read_only
                }
                for b in self.memory.blocks
            ],
            "total_chars": self.memory.total_characters(),
            "message_count": len(self.agent_state.message_ids)
        }
```

---

## Streaming Support

### Streaming Agent Implementation

```python
from typing import Generator, Dict, Any

class StreamingAgent(Agent):
    """
    Agent with streaming support for real-time responses.
    """
    
    def stream_step(
        self,
        user_message: str,
        max_chaining_steps: int = 10
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream agent responses in real-time.
        
        Yields:
            {
                "type": "thinking" | "content" | "tool_call" | "tool_result" | "done",
                "data": ...
            }
        """
        # Check context overflow
        is_overflow, _ = self.context_manager.check_context_overflow(self.agent_state)
        if is_overflow:
            yield {"type": "thinking", "data": "Summarizing conversation..."}
            self.agent_state = self.context_manager.summarize_messages_inplace(
                self.agent_state
            )
        
        # Add user message
        user_msg = {"role": "user", "content": user_message}
        self.agent_state = self.message_manager.append_to_in_context(
            self.agent_state, [user_msg]
        )
        
        step_count = 0
        while step_count < max_chaining_steps:
            step_count += 1
            
            # Build prompt
            system_prompt = PromptGenerator.build_system_message(
                agent_state=self.agent_state,
                memory=self.memory,
                message_manager=self.message_manager,
                archival_manager=self.archival_manager
            )
            
            in_context_messages = self.message_manager.get_messages_by_ids(
                self.agent_state.message_ids,
                self.agent_state.id
            )
            
            llm_messages = [
                {"role": "system", "content": system_prompt},
                *[self._format_for_llm(m) for m in in_context_messages[1:]]
            ]
            
            # Stream LLM response
            full_content = ""
            tool_calls = []
            
            stream = self.llm_client.chat.completions.create(
                model=self.agent_state.llm_config.model,
                messages=llm_messages,
                tools=get_memory_tool_schemas(),
                stream=True
            )
            
            for chunk in stream:
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                
                # Stream content
                if delta.content:
                    full_content += delta.content
                    yield {"type": "content", "data": delta.content}
                
                # Accumulate tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        while len(tool_calls) <= tc.index:
                            tool_calls.append({
                                "id": "",
                                "function": {"name": "", "arguments": ""}
                            })
                        
                        if tc.id:
                            tool_calls[tc.index]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls[tc.index]["function"]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments
            
            # Save assistant message
            assistant_msg = {"role": "assistant", "content": full_content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            
            self.agent_state = self.message_manager.append_to_in_context(
                self.agent_state, [assistant_msg]
            )
            
            # If no tool calls, done
            if not tool_calls:
                yield {"type": "done", "data": {"step_count": step_count}}
                break
            
            # Execute tools
            heartbeat = False
            for tool_call in tool_calls:
                yield {
                    "type": "tool_call",
                    "data": {
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"]["arguments"]
                    }
                }
                
                tool_msg, is_heartbeat, result = self._execute_tool(tool_call)
                
                yield {
                    "type": "tool_result",
                    "data": {
                        "name": tool_call["function"]["name"],
                        "result": tool_msg["content"][:200]
                    }
                }
                
                self.agent_state = self.message_manager.append_to_in_context(
                    self.agent_state, [tool_msg]
                )
                
                if is_heartbeat:
                    heartbeat = True
            
            if not heartbeat:
                yield {"type": "done", "data": {"step_count": step_count}}
                break
        
        self._save_state()
```

---

## Tool Rules System

### Tool Rules - Controlling Tool Execution Order

```python
from dataclasses import dataclass
from typing import List, Set, Optional
from enum import Enum

class ToolRuleType(str, Enum):
    INIT = "InitToolRule"           # Must run first
    TERMINAL = "TerminalToolRule"   # Ends execution
    CONTINUE = "ContinueToolRule"   # Forces continuation
    CHILD = "ChildToolRule"         # Must follow parent
    REQUIRES_APPROVAL = "RequiresApprovalToolRule"


@dataclass
class ToolRule:
    """Base tool rule"""
    tool_name: str
    type: ToolRuleType


@dataclass
class InitToolRule(ToolRule):
    """Tool that must be called first"""
    type: ToolRuleType = ToolRuleType.INIT


@dataclass
class TerminalToolRule(ToolRule):
    """Tool that ends the agent loop"""
    type: ToolRuleType = ToolRuleType.TERMINAL


@dataclass
class ContinueToolRule(ToolRule):
    """Tool that forces a follow-up step"""
    type: ToolRuleType = ToolRuleType.CONTINUE


@dataclass
class ChildToolRule(ToolRule):
    """Tool that must follow a specific parent tool"""
    type: ToolRuleType = ToolRuleType.CHILD
    children: List[str] = None  # Tools that can follow this one


class ToolRulesSolver:
    """
    Validates and enforces tool rules during agent execution.
    
    Letta uses tool rules to:
    - Force specific tools to run first (init rules)
    - End execution after certain tools (terminal rules)
    - Require follow-up after certain tools (continue rules)
    - Define tool execution order (child rules)
    """
    
    def __init__(self, tool_rules: Optional[List[ToolRule]] = None):
        self.tool_rules = tool_rules or []
        self.tool_call_history: List[str] = []
        
        # Parse rules by type
        self.init_tool_rules = [r for r in self.tool_rules if r.type == ToolRuleType.INIT]
        self.terminal_tool_rules = [r for r in self.tool_rules if r.type == ToolRuleType.TERMINAL]
        self.continue_tool_rules = [r for r in self.tool_rules if r.type == ToolRuleType.CONTINUE]
        self.child_tool_rules = [r for r in self.tool_rules if r.type == ToolRuleType.CHILD]
    
    def get_allowed_tool_names(
        self,
        available_tools: Set[str],
        last_function_response: Optional[str] = None
    ) -> Optional[List[str]]:
        """
        Get list of allowed tools based on current state and rules.
        
        Returns None if all tools are allowed.
        """
        # If no rules, all tools allowed
        if not self.tool_rules:
            return None
        
        # Init rule: first call must be init tool
        if not self.tool_call_history and self.init_tool_rules:
            return [r.tool_name for r in self.init_tool_rules]
        
        # Child rules: if last tool has children, only allow children
        if self.tool_call_history:
            last_tool = self.tool_call_history[-1]
            for rule in self.child_tool_rules:
                if rule.tool_name == last_tool and rule.children:
                    return [t for t in rule.children if t in available_tools]
        
        return None  # All tools allowed
    
    def register_tool_call(self, tool_name: Optional[str]):
        """Record a tool call in history"""
        if tool_name:
            self.tool_call_history.append(tool_name)
    
    def is_terminal_tool(self, tool_name: str) -> bool:
        """Check if tool is terminal (ends execution)"""
        return any(r.tool_name == tool_name for r in self.terminal_tool_rules)
    
    def is_continue_tool(self, tool_name: str) -> bool:
        """Check if tool forces continuation"""
        return any(r.tool_name == tool_name for r in self.continue_tool_rules)
    
    def has_children_tools(self, tool_name: str) -> bool:
        """Check if tool has child tools that must follow"""
        for rule in self.child_tool_rules:
            if rule.tool_name == tool_name and rule.children:
                return True
        return False
    
    def clear_tool_history(self):
        """Clear tool call history (call at start of each step)"""
        self.tool_call_history = []


# Example usage:
DEFAULT_TOOL_RULES = [
    # send_message is terminal - ends the agent loop
    TerminalToolRule(tool_name="send_message"),
    
    # Memory tools force continuation
    ContinueToolRule(tool_name="core_memory_append"),
    ContinueToolRule(tool_name="core_memory_replace"),
    ContinueToolRule(tool_name="memory_rethink"),
]
```

---

## Persistence Layer

### Database Schema & Storage

```python
import sqlite3
import json
from typing import Optional, List
from contextlib import contextmanager

class MemoryDatabase:
    """
    Persistence layer for memory blocks, messages, and agent state.
    Uses SQLite for simplicity, but can be adapted to PostgreSQL.
    """
    
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create database tables"""
        with self._get_connection() as conn:
            # Agents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    system TEXT,
                    agent_type TEXT,
                    block_ids TEXT,
                    message_ids TEXT,
                    tool_ids TEXT,
                    tool_rules TEXT,
                    llm_config TEXT,
                    embedding_config TEXT,
                    message_buffer_autoclear BOOLEAN DEFAULT 0,
                    enable_sleeptime BOOLEAN DEFAULT 0,
                    metadata TEXT,
                    tags TEXT,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    created_by_id TEXT
                )
            """)
            
            # Memory blocks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS blocks (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    value TEXT NOT NULL,
                    description TEXT,
                    char_limit INTEGER NOT NULL,
                    read_only BOOLEAN NOT NULL DEFAULT 0,
                    is_template BOOLEAN NOT NULL DEFAULT 0,
                    template_name TEXT,
                    project_id TEXT,
                    tags TEXT,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    created_by_id TEXT,
                    last_updated_by_id TEXT
                )
            """)
            
            # Messages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls TEXT,
                    tool_call_id TEXT,
                    name TEXT,
                    model TEXT,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
                )
            """)
            
            # Archival memory table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS archival_memory (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT,
                    metadata TEXT,
                    embedding TEXT,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_agent ON messages(agent_id, created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_archival_agent ON archival_memory(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_blocks_label ON blocks(label)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # ===== AGENT STATE OPERATIONS =====
    
    def save_agent_state(self, state: AgentState):
        """Save agent state"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO agents (
                    id, name, description, system, agent_type,
                    block_ids, message_ids, tool_ids, tool_rules,
                    llm_config, embedding_config,
                    message_buffer_autoclear, enable_sleeptime,
                    metadata, tags, created_at, updated_at, created_by_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                state.id,
                state.name,
                state.description,
                state.system,
                state.agent_type,
                json.dumps(state.block_ids),
                json.dumps(state.message_ids),
                json.dumps(state.tool_ids),
                json.dumps(state.tool_rules) if state.tool_rules else None,
                json.dumps(state.llm_config.to_dict()) if state.llm_config else None,
                json.dumps(state.embedding_config.to_dict()) if state.embedding_config else None,
                state.message_buffer_autoclear,
                state.enable_sleeptime,
                json.dumps(state.metadata) if state.metadata else None,
                json.dumps(state.tags),
                state.created_at.isoformat(),
                state.updated_at.isoformat(),
                state.created_by_id
            ))
            conn.commit()
    
    def load_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Load agent state by ID"""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return AgentState(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                system=row["system"],
                agent_type=row["agent_type"],
                block_ids=json.loads(row["block_ids"]) if row["block_ids"] else [],
                message_ids=json.loads(row["message_ids"]) if row["message_ids"] else [],
                tool_ids=json.loads(row["tool_ids"]) if row["tool_ids"] else [],
                tool_rules=json.loads(row["tool_rules"]) if row["tool_rules"] else None,
                llm_config=LLMConfig.from_dict(json.loads(row["llm_config"])) if row["llm_config"] else LLMConfig(),
                embedding_config=EmbeddingConfig.from_dict(json.loads(row["embedding_config"])) if row["embedding_config"] else EmbeddingConfig(),
                message_buffer_autoclear=bool(row["message_buffer_autoclear"]),
                enable_sleeptime=bool(row["enable_sleeptime"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                tags=json.loads(row["tags"]) if row["tags"] else [],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                created_by_id=row["created_by_id"]
            )
    
    # ===== BLOCK OPERATIONS =====
    
    def save_block(self, block: Block):
        """Save or update a memory block"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO blocks (
                    id, label, value, description, char_limit, read_only,
                    is_template, template_name, project_id, tags,
                    created_at, updated_at, created_by_id, last_updated_by_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                block.id,
                block.label,
                block.value,
                block.description,
                block.limit,
                block.read_only,
                block.is_template,
                block.template_name,
                block.project_id,
                json.dumps(block.tags),
                block.created_at.isoformat(),
                block.updated_at.isoformat(),
                block.created_by_id,
                block.last_updated_by_id
            ))
            conn.commit()
    
    def load_block(self, block_id: str) -> Optional[Block]:
        """Load a single block by ID"""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM blocks WHERE id = ?", (block_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_block(row)
    
    def load_blocks_by_ids(self, block_ids: List[str]) -> List[Block]:
        """Load multiple blocks by IDs"""
        if not block_ids:
            return []
        
        with self._get_connection() as conn:
            placeholders = ','.join('?' * len(block_ids))
            cursor = conn.execute(
                f"SELECT * FROM blocks WHERE id IN ({placeholders})",
                block_ids
            )
            
            block_map = {row["id"]: self._row_to_block(row) for row in cursor.fetchall()}
            return [block_map[bid] for bid in block_ids if bid in block_map]
    
    def _row_to_block(self, row: sqlite3.Row) -> Block:
        """Convert database row to Block object"""
        return Block(
            id=row["id"],
            label=row["label"],
            value=row["value"],
            description=row["description"],
            limit=row["char_limit"],
            read_only=bool(row["read_only"]),
            is_template=bool(row["is_template"]),
            template_name=row["template_name"],
            project_id=row["project_id"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            created_by_id=row["created_by_id"],
            last_updated_by_id=row["last_updated_by_id"]
        )
```

---

## Error Handling & Recovery

### Robust Error Handling

```python
class LettaError(Exception):
    """Base exception for Letta errors"""
    pass


class ContextWindowExceededError(LettaError):
    """Raised when context window cannot be reduced further"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}


class MemoryLimitError(LettaError):
    """Raised when memory block limit is exceeded"""
    pass


class ToolExecutionError(LettaError):
    """Raised when a tool fails to execute"""
    def __init__(self, tool_name: str, message: str, traceback: Optional[str] = None):
        super().__init__(f"Tool '{tool_name}' failed: {message}")
        self.tool_name = tool_name
        self.traceback = traceback


class Agent:
    """Agent class continued with error handling..."""
    
    def step_with_recovery(
        self,
        user_message: str,
        max_retries: int = 3,
        max_chaining_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Execute step with automatic error recovery.
        
        Handles:
        - Context window overflow (triggers summarization)
        - Tool execution errors (retries with error context)
        - LLM API errors (exponential backoff)
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return self.step(user_message, max_chaining_steps)
                
            except ContextWindowExceededError as e:
                print(f"⚠️ Context overflow (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    # Try more aggressive summarization
                    self.agent_state = self.context_manager.summarize_messages_inplace(
                        self.agent_state,
                        num_to_summarize=len(self.agent_state.message_ids) // 2
                    )
                    continue
                else:
                    raise
                    
            except MemoryLimitError as e:
                print(f"⚠️ Memory limit exceeded: {e}")
                # Could trigger auto-summarization of memory blocks
                raise
                
            except Exception as e:
                last_error = e
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⚠️ Error (attempt {attempt + 1}): {e}")
                    print(f"   Retrying in {wait_time}s...")
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    raise
        
        raise last_error or Exception("Max retries exceeded")
```

---

## Complete Working Example

### Putting It All Together

```python
#!/usr/bin/env python3
"""
Complete example: Letta-style memory agent
"""

from openai import OpenAI
import os


def create_agent(llm_client, db_path: str = "agent_memory.db"):
    """Create a fully configured agent"""
    
    # Initialize database
    db = MemoryDatabase(db_path)
    
    # Create memory blocks
    persona_block = Block(
        label="persona",
        value="I am a helpful AI assistant who remembers our conversations.",
        description="Your personality and behavior",
        limit=2000
    )
    
    human_block = Block(
        label="human",
        value="New user - no information yet.",
        description="Information about the user",
        limit=2000
    )
    
    # Save blocks
    db.save_block(persona_block)
    db.save_block(human_block)
    
    # Create memory
    memory = Memory(blocks=[persona_block, human_block])
    
    # Create agent state
    agent_state = AgentState(
        name="MemoryAgent",
        system=PromptGenerator.get_default_system_prompt(),
        agent_type="letta_v1_agent",
        llm_config=LLMConfig(model="gpt-4o-mini", context_window=128000),
        block_ids=[persona_block.id, human_block.id]
    )
    db.save_agent_state(agent_state)
    
    # Create managers
    message_manager = MessageManager(db)
    archival_manager = ArchivalMemoryManager(db, llm_client=llm_client)
    context_manager = ContextWindowManager(
        message_manager=message_manager,
        llm_client=llm_client,
        context_window_limit=128000
    )
    
    # Create agent
    agent = Agent(
        agent_state=agent_state,
        memory=memory,
        llm_client=llm_client,
        message_manager=message_manager,
        context_manager=context_manager,
        archival_manager=archival_manager,
        db=db
    )
    
    return agent


def main():
    print("=== Letta Memory Agent Demo ===\n")
    
    # Initialize
    llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    agent = create_agent(llm_client)
    
    # Conversation
    conversations = [
        "Hi! My name is Alice and I'm a software engineer.",
        "I prefer Python and I love detailed code examples.",
        "Can you remember what my name is?",
        "What programming language do I prefer?",
    ]
    
    for user_msg in conversations:
        print(f"USER: {user_msg}")
        result = agent.step(user_msg)
        print(f"AGENT: {result['response']}\n")
    
    # Show final memory state
    print("\n=== Final Memory State ===")
    state = agent.get_memory_state()
    for block in state["blocks"]:
        print(f"\n[{block['label']}] ({block['chars']}/{block['limit']} chars)")
        print(block['value'])
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
```

---

## Testing

### Comprehensive Test Suite

```python
import pytest

class TestBlock:
    """Tests for Block class"""
    
    def test_block_validation(self):
        """Block should reject values exceeding limit"""
        with pytest.raises(ValueError, match="exceeds"):
            Block(label="test", value="x" * 100, limit=50)
    
    def test_block_requires_label(self):
        """Block should require a label"""
        with pytest.raises(ValueError, match="label"):
            Block(label="", value="test")
    
    def test_block_update_validation(self):
        """Block should validate on value update"""
        block = Block(label="test", value="short", limit=10)
        with pytest.raises(ValueError):
            block.value = "x" * 100


class TestMemory:
    """Tests for Memory class"""
    
    def test_get_block(self):
        """Should retrieve block by label"""
        block = Block(label="test", value="content")
        memory = Memory(blocks=[block])
        
        assert memory.get_block("test").value == "content"
    
    def test_get_block_not_found(self):
        """Should raise KeyError for missing block"""
        memory = Memory(blocks=[])
        
        with pytest.raises(KeyError):
            memory.get_block("nonexistent")
    
    def test_duplicate_labels(self):
        """Should reject duplicate labels"""
        block1 = Block(label="test", value="a")
        block2 = Block(label="test", value="b")
        
        with pytest.raises(ValueError, match="Duplicate"):
            Memory(blocks=[block1, block2])


class TestMemoryTools:
    """Tests for memory editing tools"""
    
    def test_core_memory_append(self):
        """Should append to block"""
        block = Block(label="test", value="Hello", limit=100)
        memory = Memory(blocks=[block])
        tools = MemoryTools(memory)
        
        tools.core_memory_append("test", " World")
        assert "World" in memory.get_block("test").value
    
    def test_core_memory_replace(self):
        """Should replace text in block"""
        block = Block(label="test", value="Hello World", limit=100)
        memory = Memory(blocks=[block])
        tools = MemoryTools(memory)
        
        tools.core_memory_replace("test", "World", "Letta")
        assert memory.get_block("test").value == "Hello Letta"
    
    def test_read_only_protection(self):
        """Should reject edits to read-only blocks"""
        block = Block(label="test", value="Protected", read_only=True)
        memory = Memory(blocks=[block])
        tools = MemoryTools(memory)
        
        with pytest.raises(ValueError, match="read-only"):
            tools.core_memory_append("test", " data")


class TestContextWindow:
    """Tests for context window management"""
    
    def test_token_counting(self):
        """Should count tokens accurately"""
        # Setup would need actual LLM client mock
        pass
    
    def test_overflow_detection(self):
        """Should detect context overflow"""
        # Setup would need actual components
        pass


# Run with: pytest -v test_memory.py
```

---

## Production Deployment

### Deployment Checklist

```markdown
## Pre-Deployment Checklist

### Database
- [ ] Use PostgreSQL for production (not SQLite)
- [ ] Enable connection pooling
- [ ] Set up database backups
- [ ] Configure appropriate indexes

### Vector Database
- [ ] Set up Pinecone/Weaviate/pgvector for archival memory
- [ ] Configure embedding model and dimensions
- [ ] Set up index with appropriate metric (cosine)

### API Keys & Security
- [ ] Store API keys in environment variables or secrets manager
- [ ] Enable encryption for sensitive data
- [ ] Set up rate limiting
- [ ] Configure authentication

### Monitoring
- [ ] Enable logging (structured JSON recommended)
- [ ] Set up OpenTelemetry tracing
- [ ] Monitor token usage and costs
- [ ] Track memory block utilization

### Scaling
- [ ] Configure context window limits appropriately
- [ ] Set up automatic summarization thresholds
- [ ] Consider message buffer autoclear for stateless use cases
```

### Docker Compose Example

```yaml
version: '3.8'

services:
  letta-agent:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://letta:letta@postgres:5432/letta
      - PINECONE_API_KEY=${PINECONE_API_KEY}
    depends_on:
      - postgres
    ports:
      - "8000:8000"

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_USER=letta
      - POSTGRES_PASSWORD=letta
      - POSTGRES_DB=letta
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

---

## Best Practices

### 1. Memory Block Design

```python
# ✅ GOOD: Clear, focused blocks
Block(
    label="user_profile",
    value="Name: Alice\nRole: Engineer\nPreferences: Python, detailed examples",
    description="Key information about the user"
)

# ❌ BAD: Vague blocks
Block(
    label="stuff",
    value="various things...",
    description="miscellaneous"
)
```

### 2. Tool Selection Guide

| Situation | Tool | Example |
|-----------|------|---------|
| Add new fact | `core_memory_append` | Learning user's name |
| Update single fact | `core_memory_replace` | Correcting a preference |
| Major reorganization | `memory_rethink` | Restructuring all user info |
| Store for later | `archival_memory_insert` | Meeting notes, detailed specs |
| Find past info | `archival_memory_search` | "What did we discuss about X?" |

### 3. System Prompt Guidelines

```python
# Include these sections in your system prompt:
"""
1. Memory system explanation (what blocks are, how to use tools)
2. When to update memory (new facts, corrections, not casual chat)
3. How to keep memory organized (concise, structured, current)
4. Tool usage guidelines (which tool for what situation)
"""
```

### 4. Context Window Management

```python
# Set appropriate thresholds
MEMORY_WARNING_THRESHOLD = 0.75  # Warn at 75% full
SUMMARIZATION_TRIGGER = 0.85     # Summarize at 85% full
DESIRED_PRESSURE = 0.30          # Target 30% after summarization
```

---

## How to Replicate in Your Agent Framework

This section gives a concrete checklist and mapping so you can implement Letta-style memory in another agent framework.

### Mapping: Letta Concept → Your Framework

| Letta concept | What it is | What you need in your framework |
|---------------|------------|----------------------------------|
| **Block** | One named, size-limited, editable section of core memory (e.g. persona, human). | A type with: `id`, `label`, `value` (string), `limit` (char limit), `description`, optional `read_only`. Validate `len(value) <= limit` on set. |
| **Memory** | Container of blocks; compiles to XML (or similar) for the prompt. | A container that holds a list of blocks and exposes: `get_block(label)`, `update_block_value(label, value)`, `list_block_labels()`, `compile(...)` → string. |
| **AgentState** | Serializable snapshot: system prompt text, block IDs, message IDs, tools, config. | A serializable struct with: system prompt, list of block refs (or block IDs), list of message IDs (order = conversation order), tool list, LLM/config. |
| **In-context messages** | The exact messages sent to the LLM: system + ordered conversation. | Store “current context” as: one system message (or its ID) + ordered list of message IDs. Load by IDs to get full messages. |
| **System message content** | Base prompt + compiled memory + memory metadata. | Build one string: `base_prompt.replace("{CORE_MEMORY}", compiled_memory + metadata)`. Store as the content of the “system” message. |
| **Core memory tools** | `core_memory_append`, `core_memory_replace` (and variants). | Tools that: read block by label, modify value (append or replace), call `update_block_value`, then **persist and rebuild system message**. |
| **Persistence after edit** | Block value → DB; then rebuild system message (same message ID, new content). | On any core memory tool: (1) save changed block values to your store, (2) recompile memory, (3) update system message content in place. |
| **Recall (conversation history)** | All messages for the agent; “in-context” = subset by message_ids. | Full history in a message store; agent keeps `message_ids` for “current context”. Append new messages to store and to `message_ids`. Trim by dropping IDs and optionally summarizing. |
| **Archival memory** | Vector store of passages; insert and search by embedding. | Optional: embedding model, vector DB, `archival_memory_insert(text, tags)` and `archival_memory_search(query, filters)`. Update “X memories in archival” in system metadata. |
| **Summarization** | When context overflows, summarize oldest messages, trim, prepend summary message. | Optional: token counter, overflow threshold, summarizer (LLM or service), then: trim in-context message IDs, prepend one summary message, update `message_ids`. |

### Minimal Implementation Checklist

**Phase 1: Core memory only (no recall/archival)**

1. [ ] **Block type**: `id`, `label`, `value`, `limit`, `description`, `read_only`; validate length on set.
2. [ ] **Memory type**: list of blocks; `get_block(label)`, `update_block_value(label, value)`, `compile()` → string (e.g. XML).
3. [ ] **Compile format**: e.g. `<memory_blocks>` with per-block `<label>`, `<description>`, `<metadata>`, `<value>`.
4. [ ] **Agent state**: holds `memory` (or block IDs + loaded blocks), `system` (base prompt with `{CORE_MEMORY}`).
5. [ ] **System message build**: `compiled = memory.compile()`; `metadata = memory_metadata_block(...)`; `full = system.replace("{CORE_MEMORY}", compiled + "\n\n" + metadata)`.
6. [ ] **Tools**: `core_memory_append(label, content)` and `core_memory_replace(label, old, new)` that update memory then call your “persist and rebuild” function.
7. [ ] **Persist and rebuild**: on memory change, save block values to DB; reload blocks if needed; set system message content to new `full` string (same message ID if you store system as a message).

**Phase 2: In-context messages (recall)**

8. [ ] **Message store**: create/get messages by ID; order by your “context” list.
9. [ ] **Agent state**: `message_ids` = ordered list (first = system, rest = conversation).
10. [ ] **Load context**: `get_messages_by_ids(message_ids)` → list of messages in order.
11. [ ] **Append**: create new messages, append their IDs to `message_ids`, save agent state.
12. [ ] **Trim**: e.g. keep `message_ids[0]` + drop first N of the rest (for summarization).

**Phase 3: Context window and summarization (optional)**

13. [ ] **Token counting**: for system + messages (and tools if needed).
14. [ ] **Overflow check**: e.g. total tokens &gt; 0.85 × context_window → trigger summarization.
15. [ ] **Summarize**: take messages for IDs you’ll trim, call summarizer, create one “summary” message, trim those IDs, prepend summary ID to `message_ids`.

**Phase 4: Archival memory (optional)**

16. [ ] **Passage store**: embed text, store with optional tags.
17. [ ] **Tools**: `archival_memory_insert(content, tags)`, `archival_memory_search(query, tags, ...)`.
18. [ ] **Metadata**: in system message, add line like “X memories in archival (use tools to access)”.

### Integration Points in Your Agent Loop

- **Before each LLM call**: Build system message from current memory (and metadata); load in-context messages by `message_ids`; optionally check overflow and run summarization first.
- **After each LLM response**: Append assistant (and any tool) messages to your store and to `message_ids`.
- **In tool execution**: For core memory tools, update `agent_state.memory`, then run your “persist blocks + rebuild system message” once. For archival tools, call your passage insert/search and optionally force-rebuild system message so metadata count updates.

### Pitfalls to Avoid

1. **Don’t rebuild system message on every step** – only when core memory (blocks) or archival stats change.
2. **Don’t store full message content in agent state** – store message IDs and load by ID so you can trim/prepend without moving large payloads.
3. **Keep system message as one message** – same ID, replace content when memory changes; avoid creating a new system message per edit (or you’ll bloat recall).
4. **Enforce block character limits** – in block setter and in tool logic, so the LLM can’t overflow the context with one tool call.
5. **Use exact-match replace** – `core_memory_replace` requires `old_content` to appear exactly once (or document multi-match behavior); prevents accidental wipe of large sections.

---

## Summary

This implementation guide covers the complete Letta memory system:

### Core Components
- **Block**: Individual memory units with labels, values, and constraints
- **Memory**: Container managing multiple blocks with XML compilation
- **AgentState**: Serializable state separate from runtime Agent
- **MessageManager**: Production message handling with persistence
- **ContextWindowManager**: Automatic summarization on overflow
- **ArchivalMemoryManager**: Vector search for long-term memory
- **MemoryTools**: Self-editing tools for the agent
- **ToolRulesSolver**: Tool execution order enforcement

### Key Features
1. **Self-Editing Memory**: Agents modify their own memory via tools
2. **State Separation**: Clean distinction between Agent and AgentState
3. **Automatic Summarization**: Handles infinite conversations
4. **Vector Search**: Semantic search over archival memory
5. **Streaming Support**: Real-time response streaming
6. **Tool Rules**: Controlled tool execution order
7. **Error Recovery**: Robust error handling with retries

### Production Ready
- Database persistence (SQLite/PostgreSQL)
- Vector database integration
- Comprehensive testing
- Deployment configurations

---

**References:**
- Original Letta Repository: https://github.com/letta-ai/letta
- MemGPT Paper: https://arxiv.org/abs/2310.08560
- Letta Documentation: https://docs.letta.com
