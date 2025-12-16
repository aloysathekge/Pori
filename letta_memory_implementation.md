# Letta Memory System - Complete Implementation Guide

> **Based on Letta OSS (formerly MemGPT)**  
> A comprehensive guide to implementing Letta's innovative memory architecture in your own AI agent

---

## Table of Contents

1. [Overview](#overview)
2. [Core Data Models](#core-data-models)
3. [Memory Compilation & Rendering](#memory-compilation--rendering)
4. [Memory Editing Tools](#memory-editing-tools)
5. [Agent Integration](#agent-integration)
6. [Persistence Layer](#persistence-layer)
7. [Advanced Features](#advanced-features)
8. [Complete Working Example](#complete-working-example)
9. [Best Practices](#best-practices)

---

## Overview

### The Three-Tier Memory Architecture

Letta implements a hierarchical memory system inspired by computer operating systems:

```
┌─────────────────────────────────────────────────┐
│         CORE MEMORY (In-Context)                │
│  ┌──────────────┐  ┌──────────────┐            │
│  │   Persona    │  │    Human     │            │
│  │    Block     │  │    Block     │  [Custom]  │
│  └──────────────┘  └──────────────┘            │
│  Always visible to LLM | Editable via tools    │
└─────────────────────────────────────────────────┘
           ▲
           │ Search/Retrieve
           ▼
┌─────────────────────────────────────────────────┐
│      RECALL MEMORY (Conversation History)       │
│  Full chat history | Semantic + text search    │
└─────────────────────────────────────────────────┘
           ▲
           │ Search/Retrieve
           ▼
┌─────────────────────────────────────────────────┐
│    ARCHIVAL MEMORY (Long-term Storage)          │
│  Semantic embeddings | Vector search            │
└─────────────────────────────────────────────────┘
```

### Key Principles

1. **Self-Editing Memory**: Agents modify their own memory via tool calls
2. **Always In-Context**: Core memory is part of every system prompt
3. **Structured Format**: XML-based format for clarity
4. **Size Limits**: Character limits prevent unbounded growth
5. **Persistence**: All changes saved to database immediately

---

## Core Data Models

### 1. Block - The Fundamental Memory Unit

```python
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid

@dataclass
class Block:
    """
    A Block represents a reserved section of the agent's core memory.
    Each block has a label, value, and metadata.
    """
    # Core fields
    id: str = field(default_factory=lambda: f"block-{uuid.uuid4()}")
    label: str = ""                      # e.g., "persona", "human", "notes"
    value: str = ""                      # The actual memory content
    
    # Metadata
    description: Optional[str] = None    # What this block is for
    limit: int = 2000                    # Character limit
    read_only: bool = False              # Can agent edit this?
    
    # Template/organization fields (optional)
    is_template: bool = False
    template_name: Optional[str] = None
    project_id: Optional[str] = None
    
    # Audit fields (optional)
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
                f"Block value ({len(self.value)} chars) exceeds "
                f"limit ({self.limit} chars)"
            )
    
    def __len__(self) -> int:
        """Return character count of value"""
        return len(self.value)
    
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
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Block':
        """Create block from dictionary"""
        # Handle datetime fields
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        
        return cls(**data)
```

### 2. FileBlock - Special Blocks for Files

```python
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
```

### 3. Memory - The Container

```python
from typing import List, Dict, Optional
from io import StringIO

class Memory:
    """
    Represents the in-context memory (Core Memory) of the agent.
    Contains multiple Block objects that are always visible to the LLM.
    """
    
    def __init__(
        self, 
        blocks: List[Block] = None,
        file_blocks: List[FileBlock] = None
    ):
        self.blocks: List[Block] = blocks or []
        self.file_blocks: List[FileBlock] = file_blocks or []
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
            f"Block '{label}' not found. Available blocks: {available}"
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
            ValueError: If value exceeds block limit
            KeyError: If block not found
        """
        if not isinstance(value, str):
            raise ValueError("Block value must be a string")
        
        block = self.get_block(label)
        
        if block.read_only:
            raise ValueError(
                f"Cannot modify read-only block '{label}'"
            )
        
        if len(value) > block.limit:
            raise ValueError(
                f"Value ({len(value)} chars) exceeds block limit "
                f"({block.limit} chars)"
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
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Memory':
        """Deserialize memory from dictionary"""
        blocks = [Block.from_dict(b) for b in data.get("blocks", [])]
        file_blocks = [FileBlock.from_dict(fb) for fb in data.get("file_blocks", [])]
        return cls(blocks=blocks, file_blocks=file_blocks)
```

---

## Memory Compilation & Rendering

### The Core: Rendering Memory as Prompt

```python
class Memory:
    """Continued from above..."""
    
    def compile(
        self, 
        include_line_numbers: bool = False,
        sources: Optional[List] = None,
        max_files_open: Optional[int] = None
    ) -> str:
        """
        Compile memory blocks into a formatted string for the system prompt.
        This is the key method that makes memory visible to the LLM.
        
        Args:
            include_line_numbers: Add line numbers for precise editing
            sources: File sources/folders attached to agent
            max_files_open: Maximum number of files that can be open
            
        Returns:
            Formatted XML string containing all memory blocks
        """
        s = StringIO()
        
        # Render standard memory blocks
        if include_line_numbers:
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
            
            # Add warning about line numbers
            s.write("<warning>\n")
            s.write("NOTE: Line numbers shown below (with arrows like '1→') are to help during editing. ")
            s.write("Do NOT include line number prefixes in your memory edit tool calls.\n")
            s.write("</warning>\n")
            
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
    
    def _render_directories(self, s: StringIO, sources: List, max_files_open: Optional[int]):
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
            source_id = getattr(source, "id", None)
            
            s.write(f'<directory name="{source_name}">\n')
            
            if source_desc:
                s.write(f"<description>{source_desc}</description>\n")
            
            # List files in this directory
            for fb in self.file_blocks:
                if fb.source_id == source_id:
                    status = "open" if fb.is_open else "closed"
                    label = fb.label
                    
                    s.write(f'<file status="{status}" name="{label}">\n')
                    
                    if fb.description:
                        s.write(f"<description>{fb.description}</description>\n")
                    
                    if fb.is_open and fb.value:
                        s.write("<value>\n")
                        s.write(f"{fb.value}\n")
                        s.write("</value>\n")
                    
                    s.write("</file>\n")
            
            s.write("</directory>\n")
        
        s.write("</directories>")


# Example output of compile():
"""
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
from typing import Optional, Literal
import re

# Constants
LINE_NUMBER_REGEX = re.compile(r"\n\d+→ ")

class MemoryTools:
    """
    Collection of tools that allow the agent to edit its own memory.
    These are exposed to the LLM via function calling.
    """
    
    def __init__(self, memory: Memory):
        self.memory = memory
    
    # ===== BASIC TOOLS (Legacy, simpler) =====
    
    def core_memory_append(self, label: str, content: str) -> Optional[str]:
        """
        Append text to the end of a memory block.
        
        Args:
            label: The memory block to edit (e.g., "human", "persona")
            content: Text to append to the block
            
        Returns:
            None (no message returned to LLM)
            
        Example:
            core_memory_append(label="human", content="Alice prefers dark mode.")
        """
        block = self.memory.get_block(label)
        
        if block.read_only:
            raise ValueError(f"Block '{label}' is read-only and cannot be edited")
        
        current_value = block.value
        new_value = current_value + "\n" + content
        
        if len(new_value) > block.limit:
            raise ValueError(
                f"Appending would exceed {block.limit} character limit "
                f"(current: {len(current_value)}, adding: {len(content)})"
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
        
        current_value = block.value
        
        if old_content not in current_value:
            raise ValueError(
                f"Text '{old_content}' not found in memory block '{label}'"
            )
        
        new_value = current_value.replace(old_content, new_content)
        
        if len(new_value) > block.limit:
            raise ValueError(
                f"Replacement would exceed {block.limit} character limit"
            )
        
        self.memory.update_block_value(label, new_value)
        return None
    
    # ===== ADVANCED TOOLS (Precise editing) =====
    
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
            ValueError: If old_str not found or not unique
        """
        block = self.memory.get_block(label)
        
        if block.read_only:
            raise ValueError(f"Block '{label}' is read-only")
        
        # Validate no line numbers in arguments (common error)
        if LINE_NUMBER_REGEX.search(old_str):
            raise ValueError(
                "old_str contains line number prefix. "
                "Do not include line numbers - they are for display only."
            )
        
        if LINE_NUMBER_REGEX.search(new_str):
            raise ValueError(
                "new_str contains line number prefix. "
                "Do not include line numbers - they are for display only."
            )
        
        # Expand tabs for consistent matching
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs()
        current_value = block.value.expandtabs()
        
        # Check if old_str is unique
        occurrences = current_value.count(old_str)
        
        if occurrences == 0:
            raise ValueError(
                f"Text '{old_str}' not found in memory block '{label}'"
            )
        elif occurrences > 1:
            # Find line numbers where it appears
            lines = [
                idx + 1 
                for idx, line in enumerate(current_value.split("\n"))
                if old_str in line
            ]
            raise ValueError(
                f"Text '{old_str}' appears {occurrences} times in block '{label}' "
                f"(lines: {lines}). Please ensure the text is unique."
            )
        
        # Perform replacement
        new_value = current_value.replace(old_str, new_str)
        
        if len(new_value) > block.limit:
            raise ValueError(
                f"Replacement would result in {len(new_value)} characters, "
                f"exceeding the {block.limit} character limit"
            )
        
        self.memory.update_block_value(label, new_value)
        
        return (
            f"Successfully edited memory block '{label}'. "
            f"Review the changes and edit again if needed."
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
                f"Insertion would result in {len(new_value)} characters, "
                f"exceeding the {block.limit} character limit"
            )
        
        self.memory.update_block_value(label, new_value)
        
        return (
            f"Successfully inserted text into memory block '{label}'. "
            f"Review the changes and edit again if needed."
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
                - Step-by-step explanations
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
            new_block = Block(label=label, value="")
            self.memory.set_block(new_block)
        
        if len(new_memory) > self.memory.get_block(label).limit:
            raise ValueError(
                f"New memory ({len(new_memory)} chars) exceeds "
                f"limit ({self.memory.get_block(label).limit} chars)"
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
            Confirmation message
            
        Example:
            archival_memory_insert(
                content="Meeting 2024-03-15: Decided to use Python for backend",
                tags=["meetings", "decisions", "backend"]
            )
        """
        # This would integrate with your vector database
        # Implementation depends on your storage backend
        raise NotImplementedError("Archival memory requires vector storage")
    
    def archival_memory_search(
        self, 
        query: str, 
        tags: Optional[List[str]] = None,
        top_k: int = 5
    ) -> str:
        """
        Search archival memory using semantic similarity.
        
        Args:
            query: What to search for (natural language)
            tags: Filter results by tags
            top_k: Maximum results to return
            
        Returns:
            JSON string with search results
            
        Example:
            archival_memory_search(
                query="backend technology decisions",
                tags=["decisions"],
                top_k=3
            )
        """
        # This would query your vector database
        raise NotImplementedError("Archival memory requires vector storage")
    
    def conversation_search(
        self,
        query: str,
        roles: Optional[List[Literal["user", "assistant", "tool"]]] = None,
        limit: int = 10
    ) -> str:
        """
        Search past conversation history.
        
        Args:
            query: What to search for
            roles: Filter by message roles
            limit: Maximum results
            
        Returns:
            JSON string with search results including timestamps
            
        Example:
            conversation_search(
                query="Python examples",
                roles=["assistant"],
                limit=5
            )
        """
        # This would search your message database
        raise NotImplementedError("Conversation search requires message storage")


# ===== TOOL SCHEMAS FOR LLM =====

def get_memory_tool_schemas() -> List[dict]:
    """
    Return OpenAI-format function schemas for memory tools.
    These are passed to the LLM to enable function calling.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "core_memory_append",
                "description": "Append content to a memory block",
                "parameters": {
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
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "core_memory_replace",
                "description": "Replace specific text in a memory block",
                "parameters": {
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
                            "description": "Text to replace with (empty string to delete)",
                        },
                    },
                    "required": ["label", "old_content", "new_content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_rethink",
                "description": "Completely rewrite a memory block (for major reorganization)",
                "parameters": {
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
                },
            },
        },
    ]
```

---

## Agent Integration

### Building the System Prompt

```python
def build_system_prompt(memory: Memory, agent_type: str = "default") -> str:
    """
    Build the complete system prompt including memory blocks.
    
    Args:
        memory: The Memory object containing blocks
        agent_type: Type of agent (affects prompt style)
        
    Returns:
        Complete system prompt string
    """
    
    # Base instructions
    base_instructions = """You are a helpful AI assistant with advanced memory capabilities.

<memory>
You have an advanced memory system that enables you to remember past interactions.
Your memory consists of memory blocks and external memory:

- Memory Blocks: Stored as memory blocks, each containing a label (title), 
  description (explaining how this block should influence your behavior), 
  and value (the actual content). Memory blocks have size limits and are 
  embedded within your system instructions, remaining constantly available in-context.

- External memory: Additional memory storage that is accessible via tools
  when needed (conversation search, archival search).

Memory management tools allow you to edit existing memory blocks to remember 
important information about the user and adapt your behavior over time.
</memory>

<memory_guidelines>
- Update memory blocks when you learn important new information about the user
- Keep memory organized, clear, and concise
- Remove outdated information when updating
- Use core_memory_replace for precise edits
- Use memory_rethink only for major reorganization
</memory_guidelines>
"""
    
    # Compile memory blocks into XML format
    memory_str = memory.compile(
        include_line_numbers=(agent_type == "sleeptime")
    )
    
    # Combine everything
    full_prompt = f"{base_instructions}\n\n{memory_str}"
    
    return full_prompt
```

### Agent Loop with Memory Integration

```python
from typing import List, Dict, Any
import json

class MemoryAgent:
    """
    An agent with Letta-style self-editing memory.
    Integrates memory blocks with LLM function calling.
    """
    
    def __init__(
        self,
        memory: Memory,
        llm_client: Any,  # Your LLM client (OpenAI, Anthropic, etc.)
        model: str = "gpt-4",
        agent_type: str = "default"
    ):
        self.memory = memory
        self.llm_client = llm_client
        self.model = model
        self.agent_type = agent_type
        
        # Initialize memory tools
        self.memory_tools = MemoryTools(memory)
        
        # Message history
        self.messages: List[Dict[str, str]] = []
        
        # Tool registry
        self.tools = {
            "core_memory_append": self.memory_tools.core_memory_append,
            "core_memory_replace": self.memory_tools.core_memory_replace,
            "memory_replace": self.memory_tools.memory_replace,
            "memory_insert": self.memory_tools.memory_insert,
            "memory_rethink": self.memory_tools.memory_rethink,
        }
    
    def step(self, user_message: str, max_steps: int = 10) -> str:
        """
        Execute one turn of the agent loop.
        Handles tool calls including memory edits.
        
        Args:
            user_message: Message from the user
            max_steps: Maximum tool calling iterations
            
        Returns:
            Final assistant message
        """
        
        # Build system prompt with current memory state
        system_prompt = build_system_prompt(self.memory, self.agent_type)
        
        # Add user message to history
        self.messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Prepare messages for LLM
        llm_messages = [
            {"role": "system", "content": system_prompt},
            *self.messages
        ]
        
        # Tool calling loop
        steps = 0
        while steps < max_steps:
            steps += 1
            
            # Call LLM with tools
            response = self._call_llm(llm_messages)
            
            # Check if there are tool calls
            if not response.get("tool_calls"):
                # No more tool calls - return final message
                assistant_message = response.get("content", "")
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                return assistant_message
            
            # Execute tool calls
            assistant_msg = {
                "role": "assistant",
                "content": response.get("content", ""),
                "tool_calls": response["tool_calls"]
            }
            llm_messages.append(assistant_msg)
            
            for tool_call in response["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                
                # Execute the tool
                try:
                    if tool_name in self.tools:
                        result = self.tools[tool_name](**tool_args)
                        result_str = str(result) if result else "Success"
                    else:
                        result_str = f"Error: Unknown tool '{tool_name}'"
                    
                    success = True
                except Exception as e:
                    result_str = f"Error: {str(e)}"
                    success = False
                
                # Add tool result to messages
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_name,
                    "content": result_str
                }
                llm_messages.append(tool_msg)
                
                # Log tool execution
                print(f"[TOOL] {tool_name}(**{tool_args})")
                print(f"[RESULT] {result_str}\n")
        
        # Max steps reached
        last_message = llm_messages[-1].get("content", "")
        return last_message or "Max steps reached"
    
    def _call_llm(self, messages: List[Dict]) -> Dict:
        """
        Call the LLM with function calling enabled.
        
        Args:
            messages: Conversation messages
            
        Returns:
            LLM response with potential tool_calls
        """
        # Example for OpenAI
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=get_memory_tool_schemas(),
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        result = {
            "content": message.content,
            "tool_calls": []
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
    
    def get_memory_state(self) -> Dict:
        """Get current memory state for inspection"""
        return {
            "blocks": [
                {
                    "label": b.label,
                    "value": b.value,
                    "chars": len(b),
                    "limit": b.limit
                }
                for b in self.memory.blocks
            ],
            "total_chars": self.memory.total_characters()
        }
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
    Persistence layer for memory blocks and conversation history.
    Uses SQLite for simplicity, but can be adapted to PostgreSQL.
    """
    
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create database tables"""
        with self._get_connection() as conn:
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
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    created_by_id TEXT,
                    last_updated_by_id TEXT,
                    UNIQUE(label)
                )
            """)
            
            # Messages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls TEXT,  -- JSON array
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    agent_id TEXT
                )
            """)
            
            # Archival memory table (for vector storage)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS archival_memory (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding BLOB,  -- Serialized vector
                    tags TEXT,  -- JSON array
                    created_at TIMESTAMP NOT NULL,
                    agent_id TEXT
                )
            """)
            
            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_agent_timestamp 
                ON messages(agent_id, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_archival_agent 
                ON archival_memory(agent_id)
            """)
            
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
    
    # ===== BLOCK OPERATIONS =====
    
    def save_block(self, block: Block):
        """Save or update a memory block"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO blocks (
                    id, label, value, description, char_limit, read_only,
                    is_template, template_name, project_id,
                    created_at, updated_at, created_by_id, last_updated_by_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                block.created_at.isoformat(),
                block.updated_at.isoformat(),
                block.created_by_id,
                block.last_updated_by_id
            ))
            conn.commit()
    
    def load_block(self, block_id: str) -> Optional[Block]:
        """Load a single block by ID"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM blocks WHERE id = ?",
                (block_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_block(row)
    
    def load_all_blocks(self) -> List[Block]:
        """Load all memory blocks"""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM blocks ORDER BY label")
            return [self._row_to_block(row) for row in cursor.fetchall()]
    
    def load_memory(self) -> Memory:
        """Load complete Memory object with all blocks"""
        blocks = self.load_all_blocks()
        return Memory(blocks=blocks)
    
    def delete_block(self, block_id: str):
        """Delete a block by ID"""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM blocks WHERE id = ?", (block_id,))
            conn.commit()
    
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
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            created_by_id=row["created_by_id"],
            last_updated_by_id=row["last_updated_by_id"]
        )
    
    # ===== MESSAGE OPERATIONS =====
    
    def save_message(
        self,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict]] = None,
        agent_id: str = "default"
    ):
        """Save a message to conversation history"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO messages (role, content, tool_calls, agent_id)
                VALUES (?, ?, ?, ?)
            """, (
                role,
                content,
                json.dumps(tool_calls) if tool_calls else None,
                agent_id
            ))
            conn.commit()
    
    def load_messages(
        self,
        agent_id: str = "default",
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Load conversation history"""
        with self._get_connection() as conn:
            query = """
                SELECT * FROM messages 
                WHERE agent_id = ?
                ORDER BY timestamp DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor = conn.execute(query, (agent_id,))
            
            messages = []
            for row in cursor.fetchall():
                msg = {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"]
                }
                
                if row["tool_calls"]:
                    msg["tool_calls"] = json.loads(row["tool_calls"])
                
                messages.append(msg)
            
            return list(reversed(messages))  # Chronological order
    
    def search_messages(
        self,
        query: str,
        agent_id: str = "default",
        limit: int = 10
    ) -> List[Dict]:
        """
        Simple text search over messages.
        For production, use FTS5 or external search engine.
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM messages
                WHERE agent_id = ?
                AND content LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (agent_id, f"%{query}%", limit))
            
            return [
                {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"]
                }
                for row in cursor.fetchall()
            ]
    
    # ===== ARCHIVAL MEMORY OPERATIONS =====
    
    def save_archival_memory(
        self,
        memory_id: str,
        content: str,
        embedding: Optional[bytes] = None,
        tags: Optional[List[str]] = None,
        agent_id: str = "default"
    ):
        """Save to archival memory with optional embedding"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO archival_memory 
                (id, content, embedding, tags, created_at, agent_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                memory_id,
                content,
                embedding,
                json.dumps(tags) if tags else None,
                datetime.utcnow().isoformat(),
                agent_id
            ))
            conn.commit()
    
    def search_archival_memory(
        self,
        query: str,
        agent_id: str = "default",
        limit: int = 10
    ) -> List[Dict]:
        """
        Search archival memory.
        Note: This is a simple text search. For semantic search,
        integrate with a vector database (Pinecone, Weaviate, etc.)
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, content, tags, created_at
                FROM archival_memory
                WHERE agent_id = ?
                AND content LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (agent_id, f"%{query}%", limit))
            
            return [
                {
                    "id": row["id"],
                    "content": row["content"],
                    "tags": json.loads(row["tags"]) if row["tags"] else [],
                    "created_at": row["created_at"]
                }
                for row in cursor.fetchall()
            ]
```

---

## Advanced Features

### 1. Sleeptime Agents (Background Memory Management)

```python
class SleeptimeAgent:
    """
    A background agent that reorganizes and maintains memory
    while the main agent is not actively responding.
    """
    
    def __init__(
        self,
        memory: Memory,
        llm_client: Any,
        model: str = "gpt-4o-mini"  # Use cheaper model
    ):
        self.memory = memory
        self.llm_client = llm_client
        self.model = model
        self.memory_tools = MemoryTools(memory)
    
    def process_conversation_update(
        self,
        recent_messages: List[Dict],
        max_steps: int = 5
    ):
        """
        Process recent conversation and update memory accordingly.
        Called in the background after main agent responses.
        """
        
        # Build context
        context = self._build_context(recent_messages)
        
        # System prompt for sleeptime agent
        system_prompt = """You are a memory management system.
Your job is to review conversation updates and maintain organized,
comprehensive memory blocks.

Guidelines:
- Update memory blocks to reflect new information
- Remove outdated or redundant information
- Keep memory organized and readable
- Be selective - not every message needs a memory update
- Call memory_finish_edits when done
"""
        
        # Add current memory state with line numbers
        memory_str = self.memory.compile(include_line_numbers=True)
        full_prompt = f"{system_prompt}\n\n{memory_str}\n\n{context}"
        
        messages = [{"role": "system", "content": full_prompt}]
        
        # Tool calling loop
        steps = 0
        while steps < max_steps:
            steps += 1
            
            response = self._call_llm(messages)
            
            if not response.get("tool_calls"):
                break
            
            # Execute tools
            for tool_call in response["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                
                # Stop if finish_edits called
                if tool_name == "memory_finish_edits":
                    return
                
                # Execute memory tool
                tool_args = json.loads(tool_call["function"]["arguments"])
                try:
                    if hasattr(self.memory_tools, tool_name):
                        method = getattr(self.memory_tools, tool_name)
                        result = method(**tool_args)
                    else:
                        result = f"Unknown tool: {tool_name}"
                except Exception as e:
                    result = f"Error: {str(e)}"
                
                # Add result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_name,
                    "content": str(result) if result else "Success"
                })
    
    def _build_context(self, recent_messages: List[Dict]) -> str:
        """Format recent messages for sleeptime agent"""
        context = "<recent_conversation>\n"
        for msg in recent_messages[-10:]:  # Last 10 messages
            role = msg.get("role", "")
            content = msg.get("content", "")
            context += f"<{role}>\n{content}\n</{role}>\n"
        context += "</recent_conversation>"
        return context
    
    def _call_llm(self, messages: List[Dict]) -> Dict:
        """Call LLM for sleeptime processing"""
        # Similar to main agent's _call_llm
        pass
```

### 2. Multi-Agent Shared Memory

```python
class SharedMemoryBlock(Block):
    """A memory block that can be shared across multiple agents"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_with: List[str] = []  # Agent IDs
    
    def add_sharing(self, agent_id: str):
        """Share this block with another agent"""
        if agent_id not in self.shared_with:
            self.shared_with.append(agent_id)
    
    def remove_sharing(self, agent_id: str):
        """Stop sharing with an agent"""
        if agent_id in self.shared_with:
            self.shared_with.remove(agent_id)


def create_multi_agent_system(
    num_agents: int,
    shared_block: SharedMemoryBlock,
    llm_client: Any
) -> List[MemoryAgent]:
    """
    Create multiple agents sharing a common memory block.
    
    Example use case:
    - Supervisor agent + worker agents sharing project context
    - Multiple specialists sharing user preferences
    """
    agents = []
    
    for i in range(num_agents):
        # Each agent has its own persona
        persona_block = Block(
            label="persona",
            value=f"I am agent {i} in a multi-agent system.",
            description="My role and personality"
        )
        
        # But they all share a common block
        memory = Memory(blocks=[persona_block, shared_block])
        
        agent = MemoryAgent(
            memory=memory,
            llm_client=llm_client,
            agent_type=f"agent_{i}"
        )
        
        shared_block.add_sharing(agent.agent_type)
        agents.append(agent)
    
    return agents
```

### 3. Memory Summarization

```python
def summarize_memory_block(
    block: Block,
    llm_client: Any,
    target_length: Optional[int] = None
) -> str:
    """
    Automatically summarize a memory block when it gets too large.
    
    Args:
        block: Block to summarize
        llm_client: LLM client
        target_length: Target character count (default: 80% of limit)
        
    Returns:
        Summarized content
    """
    if target_length is None:
        target_length = int(block.limit * 0.8)
    
    prompt = f"""Summarize the following memory block, preserving key information.

Target length: ~{target_length} characters
Current content:

{block.value}

Provide a concise summary that:
1. Keeps all critical facts
2. Removes redundancy
3. Uses clear, efficient language
4. Maintains chronological context

Summary:"""
    
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    summary = response.choices[0].message.content
    return summary.strip()


def auto_maintain_memory(memory: Memory, llm_client: Any):
    """
    Automatically maintain memory blocks when they approach limits.
    """
    for block in memory.blocks:
        if block.read_only:
            continue
        
        # If block is >90% full, summarize it
        usage = len(block) / block.limit
        if usage > 0.9:
            print(f"Block '{block.label}' is {usage:.0%} full. Summarizing...")
            
            summarized = summarize_memory_block(block, llm_client)
            memory.update_block_value(block.label, summarized)
            
            print(f"Reduced from {len(block.value)} to {len(summarized)} chars")
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

def main():
    # 1. Initialize components
    print("=== Initializing Letta Memory Agent ===\n")
    
    # Create memory blocks
    persona_block = Block(
        id="block-persona-001",
        label="persona",
        value="I am a helpful coding assistant who specializes in Python.",
        description="Your personality and behavior as an assistant",
        limit=2000
    )
    
    human_block = Block(
        id="block-human-001",
        label="human",
        value="New user - no information yet.",
        description="Information about the user",
        limit=2000
    )
    
    notes_block = Block(
        id="block-notes-001",
        label="notes",
        value="",
        description="Important notes and context from conversations",
        limit=3000
    )
    
    # Create memory
    memory = Memory(blocks=[persona_block, human_block, notes_block])
    
    # Initialize database
    db = MemoryDatabase("agent_demo.db")
    for block in memory.blocks:
        db.save_block(block)
    
    # Initialize agent
    llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    agent = MemoryAgent(
        memory=memory,
        llm_client=llm_client,
        model="gpt-4o-mini"
    )
    
    # 2. Conversation
    print("Starting conversation...\n")
    print("-" * 60)
    
    # Turn 1: User introduces themselves
    user_msg_1 = "Hi! My name is Alice and I'm learning about AI agents."
    print(f"USER: {user_msg_1}")
    
    response_1 = agent.step(user_msg_1)
    print(f"AGENT: {response_1}\n")
    
    # Save to database
    db.save_message("user", user_msg_1)
    db.save_message("assistant", response_1)
    
    # Check memory state
    print("\n--- Memory State After Turn 1 ---")
    memory_state = agent.get_memory_state()
    for block in memory_state["blocks"]:
        if block["label"] == "human":
            print(f"{block['label']}: {block['value'][:100]}...")
    
    print("\n" + "-" * 60 + "\n")
    
    # Turn 2: User mentions preference
    user_msg_2 = "I prefer examples with lots of comments. Also, I really like list comprehensions!"
    print(f"USER: {user_msg_2}")
    
    response_2 = agent.step(user_msg_2)
    print(f"AGENT: {response_2}\n")
    
    db.save_message("user", user_msg_2)
    db.save_message("assistant", response_2)
    
    # Turn 3: Test if agent remembers
    user_msg_3 = "Can you show me a Python example?"
    print(f"USER: {user_msg_3}")
    
    response_3 = agent.step(user_msg_3)
    print(f"AGENT: {response_3}\n")
    
    # 3. Final memory state
    print("\n" + "=" * 60)
    print("FINAL MEMORY STATE")
    print("=" * 60 + "\n")
    
    memory_state = agent.get_memory_state()
    for block_info in memory_state["blocks"]:
        print(f"Block: {block_info['label']}")
        print(f"Content:\n{block_info['value']}\n")
        print(f"Usage: {block_info['chars']}/{block_info['limit']} chars")
        print("-" * 60 + "\n")
    
    # 4. Demonstrate persistence
    print("\n=== Demonstrating Persistence ===\n")
    
    # Save all blocks
    for block in memory.blocks:
        db.save_block(block)
    
    # Reload from database
    loaded_memory = db.load_memory()
    print(f"Loaded {len(loaded_memory.blocks)} blocks from database")
    
    # Verify memory is identical
    for original, loaded in zip(memory.blocks, loaded_memory.blocks):
        assert original.label == loaded.label
        assert original.value == loaded.value
        print(f"✓ Block '{original.label}' successfully persisted")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
```

### Expected Output

```
=== Initializing Letta Memory Agent ===

Starting conversation...

------------------------------------------------------------
USER: Hi! My name is Alice and I'm learning about AI agents.
[TOOL] core_memory_replace(**{'label': 'human', 'old_content': 'New user - no information yet.', 'new_content': "User's name is Alice. She is learning about AI agents."})
[RESULT] None

AGENT: Nice to meet you, Alice! It's great that you're diving into AI agents - it's a fascinating field. I'd be happy to help you learn. What aspects of AI agents are you most interested in exploring?

--- Memory State After Turn 1 ---
human: User's name is Alice. She is learning about AI agents.

------------------------------------------------------------

USER: I prefer examples with lots of comments. Also, I really like list comprehensions!
[TOOL] core_memory_append(**{'label': 'human', 'content': 'Prefers code examples with detailed comments. Likes Python list comprehensions.'})
[RESULT] None

AGENT: Got it! I'll make sure to include plenty of comments in examples and showcase list comprehensions when appropriate. They're definitely one of Python's most elegant features!

USER: Can you show me a Python example?
AGENT: Sure! Here's an example that uses list comprehensions with detailed comments:

```python
# List of numbers we want to process
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Using list comprehension to filter even numbers and square them
# This is more concise than a traditional for loop
squared_evens = [
    num ** 2  # Square the number
    for num in numbers  # Iterate through all numbers
    if num % 2 == 0  # Only include even numbers
]

print(squared_evens)  # Output: [4, 16, 36, 64, 100]
```

============================================================
FINAL MEMORY STATE
============================================================

Block: persona
Content:
I am a helpful coding assistant who specializes in Python.

Usage: 62/2000 chars
------------------------------------------------------------

Block: human
Content:
User's name is Alice. She is learning about AI agents.
Prefers code examples with detailed comments. Likes Python list comprehensions.

Usage: 122/2000 chars
------------------------------------------------------------

Block: notes
Content:


Usage: 0/3000 chars
------------------------------------------------------------

=== Demonstrating Persistence ===

Loaded 3 blocks from database
✓ Block 'persona' successfully persisted
✓ Block 'human' successfully persisted
✓ Block 'notes' successfully persisted

=== Demo Complete ===
```

---

## Best Practices

### 1. Memory Block Design

```python
# ✅ GOOD: Clear, focused blocks
persona_block = Block(
    label="persona",
    value="Helpful coding assistant specializing in Python",
    description="Your personality and behavior"
)

# ❌ BAD: Vague, unfocused blocks
misc_block = Block(
    label="stuff",
    value="Various things about various topics",
    description="Miscellaneous information"
)
```

### 2. Memory Limits

```python
# Set appropriate limits based on use case
SMALL_BLOCK_LIMIT = 500    # For simple facts (name, preferences)
MEDIUM_BLOCK_LIMIT = 2000  # For detailed context (persona, user info)
LARGE_BLOCK_LIMIT = 5000   # For extensive notes (project details)

# Monitor usage
def check_memory_health(memory: Memory):
    """Check if any blocks are approaching limits"""
    warnings = []
    for block in memory.blocks:
        usage = len(block) / block.limit
        if usage > 0.9:
            warnings.append(f"Block '{block.label}' is {usage:.0%} full")
    return warnings
```

### 3. Tool Selection

```python
# Use the right tool for the job

# Simple append - use core_memory_append
core_memory_append(label="human", content="Prefers dark mode")

# Precise replacement - use memory_replace
memory_replace(
    label="human",
    old_str="Prefers light mode",
    new_str="Prefers dark mode"
)

# Major reorganization - use memory_rethink
memory_rethink(
    label="human",
    new_memory="""User: Alice
Role: Software Engineer
Experience: 5 years
Preferences:
- Dark mode
- Python over JavaScript
- Detailed code comments
"""
)
```

### 4. System Prompt Guidelines

```python
def create_effective_system_prompt(memory: Memory) -> str:
    """Create a system prompt that encourages good memory management"""
    
    return f"""You are an AI assistant with self-editing memory capabilities.

<memory_instructions>
Your memory blocks contain important context. Update them when:
✓ You learn new information about the user
✓ The user corrects previous information
✓ Preferences or requirements change
✓ You complete tasks worth remembering

DO NOT update memory for:
✗ Casual conversation without new information
✗ Temporary context (will be in chat history)
✗ Information already accurately recorded
</memory_instructions>

<editing_guidelines>
- Be precise: Only replace exactly what needs changing
- Be concise: Remove outdated info when adding new
- Be organized: Keep memory readable and structured
- Be selective: Not every message needs a memory edit
</editing_guidelines>

{memory.compile()}
"""
```

### 5. Testing Memory Behavior

```python
def test_memory_agent():
    """Test suite for memory agent behavior"""
    
    # Setup
    memory = Memory(blocks=[
        Block(label="human", value="User name: Unknown", limit=1000)
    ])
    
    # Test 1: Memory update
    tools = MemoryTools(memory)
    tools.core_memory_replace(
        label="human",
        old_content="Unknown",
        new_content="Alice"
    )
    assert "Alice" in memory.get_block("human").value
    
    # Test 2: Limit enforcement
    try:
        tools.core_memory_append(
            label="human",
            content="x" * 2000  # Exceeds limit
        )
        assert False, "Should have raised error"
    except ValueError as e:
        assert "limit" in str(e).lower()
    
    # Test 3: Read-only protection
    memory.get_block("human").read_only = True
    try:
        tools.core_memory_append(label="human", content="test")
        assert False, "Should have raised error"
    except ValueError as e:
        assert "read-only" in str(e).lower()
    
    print("✓ All tests passed")
```

---

## Summary

This implementation guide covers the complete Letta memory system:

### Core Components
- **Block**: Individual memory units with labels, values, and metadata
- **Memory**: Container managing multiple blocks
- **MemoryTools**: Functions for agent self-editing
- **Agent Loop**: Integration with LLM function calling

### Key Features
1. **Self-Editing**: Agents modify their own memory via tools
2. **Structured Format**: XML-based for clarity
3. **Persistence**: Database storage for durability
4. **Validation**: Character limits and constraints
5. **Flexibility**: Support for different agent types

### Advanced Capabilities
- Sleeptime agents for background maintenance
- Shared memory across multiple agents
- Automatic summarization
- Conversation history search
- Archival memory with embeddings

Use this as a foundation to build your own agent with Letta-style memory capabilities!

---

**References:**
- Original Letta Repository: https://github.com/letta-ai/letta
- MemGPT Paper: https://arxiv.org/abs/2310.08560

