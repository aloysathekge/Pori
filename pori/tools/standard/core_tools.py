"""
Core tools required for agent operation.
"""

import logging
from typing import Any, Dict

from pydantic import BaseModel, Field

from ..registry import tool_registry

Registry = tool_registry()
logger = logging.getLogger("pori.core_tools")


class AnswerParams(BaseModel):
    final_answer: str = Field(
        ..., description="The final answer to the user's question"
    )
    reasoning: str = Field(
        ..., description="Brief explanation of how you arrived at this answer"
    )


@Registry.tool(
    name="answer",
    description="REQUIRED: Provide your final answer to the user's question with reasoning.",
)
def answer_tool(params: AnswerParams, context: Dict[str, Any]) -> Dict[str, Any]:
    """Provide a final answer to the user's question."""
    answer = {
        "final_answer": params.final_answer,
        "reasoning": params.reasoning or "No additional reasoning provided.",
    }

    logger.info(f"Answer tool called with final answer: {params.final_answer}")

    if context and "memory" in context:
        context["memory"].update_state("final_answer", answer)
        logger.info("Final answer stored in memory")
    else:
        logger.warning("Could not store final answer - memory not available in context")

    return answer


class DoneParams(BaseModel):
    success: bool = Field(
        True, description="Whether the task was completed successfully"
    )
    message: str = Field(
        "Task completed", description="Final message about task completion"
    )


@Registry.tool(name="done", description="Mark the task as complete")
def done_tool(params: DoneParams, context: Dict[str, Any]):
    """Mark the task as done."""
    return {"final_message": params.message, "success": params.success}


class AskUserParams(BaseModel):
    question: str = Field(
        ..., description="The question to ask the user for clarification"
    )
    reason: str = Field(
        ...,
        description="Why you need this information before proceeding",
    )


@Registry.tool(
    name="ask_user",
    description="Ask the user a clarifying question when you lack required information to complete the task. Use this BEFORE taking action when critical details are missing.",
)
def ask_user_tool(params: AskUserParams, context: Dict[str, Any]) -> Dict[str, Any]:
    """Ask the user for clarification and return their response."""
    print(f"\n[Agent needs clarification]\n{params.question}")
    try:
        user_response = input("Your answer: ").strip()
    except EOFError:
        user_response = ""
    return {"success": True, "user_response": user_response}


class ThinkParams(BaseModel):
    thoughts: str = Field(..., description="What you're thinking about")
    next_action: str = Field(..., description="What you plan to do next")


@Registry.tool(name="think", description="Record thoughts and planning (optional)")
def think_tool(params: ThinkParams, context: Dict[str, Any]):
    """Allow the agent to record its thoughts and planning."""
    return {
        "thoughts": params.thoughts,
        "next_action": params.next_action,
        "timestamp": "thinking step recorded",
    }


class RememberParams(BaseModel):
    text: str = Field(..., description="The fact or note to remember long-term")
    importance: int = Field(
        1, ge=1, le=5, description="Importance from 1 (low) to 5 (high)"
    )


@Registry.tool(
    name="remember",
    description="Store a fact or note into long-term memory with optional importance",
)
def remember_tool(params: RememberParams, context: Dict[str, Any]):
    """Persist a fact to long-term memory (and vector index if enabled)."""
    memory = context.get("memory") if context else None
    if not memory:
        return {"success": False, "error": "Memory not available in context"}

    try:
        key = memory.add_experience(
            params.text,
            importance=params.importance,
            meta={"type": "fact", "source": "user"},
        )
        return {"success": True, "result": {"key": key}}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------- Recall + Archival memory tools (Letta-style) ----------


class ConversationSearchParams(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(10, ge=1, le=50, description="Max results to return")
    roles: list[str] | None = Field(
        default=None, description="Optional roles filter (e.g. ['user','assistant'])"
    )


@Registry.tool(
    name="conversation_search",
    description="Search the conversation history for relevant prior messages.",
)
def conversation_search_tool(params: ConversationSearchParams, context: Dict[str, Any]):
    memory = context.get("memory") if context else None
    if not memory or not hasattr(memory, "conversation_search"):
        return {"success": False, "error": "Memory search not available"}
    try:
        results = memory.conversation_search(
            query=params.query, limit=params.limit, roles=params.roles
        )
        return {"success": True, "results": results, "count": len(results)}
    except Exception as e:
        return {"success": False, "error": str(e)}


class ArchivalInsertParams(BaseModel):
    text: str = Field(..., description="Passage to store in archival memory")
    tags: list[str] | None = Field(default=None, description="Optional tags")
    importance: int = Field(1, ge=1, le=5, description="Importance from 1-5")


@Registry.tool(
    name="archival_memory_insert",
    description="Store a passage in long-term archival memory for later retrieval.",
)
def archival_memory_insert_tool(params: ArchivalInsertParams, context: Dict[str, Any]):
    memory = context.get("memory") if context else None
    if not memory or not hasattr(memory, "archival_memory_insert"):
        return {"success": False, "error": "Archival memory not available"}
    try:
        pid = memory.archival_memory_insert(
            text=params.text, tags=params.tags, importance=params.importance
        )
        return {"success": True, "result": {"id": pid}}
    except Exception as e:
        return {"success": False, "error": str(e)}


class ArchivalSearchParams(BaseModel):
    query: str = Field(..., description="Search query")
    k: int = Field(5, ge=1, le=20, description="Max results to return")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Minimum score")
    tags: list[str] | None = Field(default=None, description="Optional tag filter")


@Registry.tool(
    name="archival_memory_search",
    description="Search archival memory passages by query using semantic similarity.",
)
def archival_memory_search_tool(params: ArchivalSearchParams, context: Dict[str, Any]):
    memory = context.get("memory") if context else None
    if not memory or not hasattr(memory, "archival_memory_search"):
        return {"success": False, "error": "Archival memory not available"}
    try:
        results = memory.archival_memory_search(
            query=params.query,
            k=params.k,
            min_score=params.min_score,
            tags=params.tags,
        )
        return {
            "success": True,
            "results": [
                {"id": rid, "text": text, "score": score}
                for rid, text, score in results
            ],
            "count": len(results),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------- Letta-style core memory (editable in-context blocks) ----------


class CoreMemoryAppendParams(BaseModel):
    label: str = Field(
        ...,
        description="Block to edit: 'persona' (your traits), 'human' (user facts), or 'notes' (scratch)",
    )
    content: str = Field(..., description="Text to append to the block")


class CoreMemoryReplaceParams(BaseModel):
    label: str = Field(
        ...,
        description="Block to edit: 'persona', 'human', or 'notes'",
    )
    old_string: str = Field(..., description="Exact text to find and replace")
    new_string: str = Field(..., description="Replacement text")


class MemoryInsertParams(BaseModel):
    label: str = Field(
        ...,
        description="Block to edit: 'persona', 'human', or 'notes'",
    )
    new_str: str = Field(..., description="Text to insert")
    insert_line: int = Field(
        -1,
        description="Line index to insert at (0-based). Use -1 to append",
    )


class MemoryRethinkParams(BaseModel):
    label: str = Field(
        ...,
        description="Block to rewrite: 'persona', 'human', or 'notes'",
    )
    new_memory: str = Field(..., description="Complete new content for the block")


@Registry.tool(
    name="core_memory_append",
    description="Append a line to a core memory block (persona, human, or notes). Always in context.",
)
def core_memory_append_tool(params: CoreMemoryAppendParams, context: Dict[str, Any]):
    """Append content to a core memory block."""
    memory = context.get("memory") if context else None
    if not memory or not getattr(memory, "core_memory", None):
        return {"success": False, "error": "Core memory not available"}
    try:
        memory.core_memory.get_block(params.label).append(params.content)
        if hasattr(memory, "persist"):
            memory.persist()
        return {"success": True, "message": f"Appended to {params.label}"}
    except ValueError as e:
        return {"success": False, "error": str(e)}


@Registry.tool(
    name="core_memory_replace",
    description="Replace exact text in a core memory block (persona, human, or notes).",
)
def core_memory_replace_tool(params: CoreMemoryReplaceParams, context: Dict[str, Any]):
    """Replace text in a core memory block."""
    memory = context.get("memory") if context else None
    if not memory or not getattr(memory, "core_memory", None):
        return {"success": False, "error": "Core memory not available"}
    try:
        memory.core_memory.get_block(params.label).replace(
            params.old_string, params.new_string
        )
        if hasattr(memory, "persist"):
            memory.persist()
        return {"success": True, "message": f"Updated {params.label}"}
    except ValueError as e:
        return {"success": False, "error": str(e)}


@Registry.tool(
    name="memory_insert",
    description="Insert text into a core memory block at a specific line index.",
)
def memory_insert_tool(params: MemoryInsertParams, context: Dict[str, Any]):
    memory = context.get("memory") if context else None
    if not memory or not getattr(memory, "core_memory", None):
        return {"success": False, "error": "Core memory not available"}
    try:
        memory.core_memory.memory_insert(
            label=params.label,
            new_str=params.new_str,
            insert_line=params.insert_line,
        )
        if hasattr(memory, "persist"):
            memory.persist()
        return {"success": True, "message": f"Inserted text into {params.label}"}
    except ValueError as e:
        return {"success": False, "error": str(e)}


@Registry.tool(
    name="memory_rethink",
    description="Rewrite an entire core memory block with a validated replacement.",
)
def memory_rethink_tool(params: MemoryRethinkParams, context: Dict[str, Any]):
    memory = context.get("memory") if context else None
    if not memory or not getattr(memory, "core_memory", None):
        return {"success": False, "error": "Core memory not available"}
    try:
        memory.core_memory.memory_rethink(
            label=params.label,
            new_memory=params.new_memory,
        )
        if hasattr(memory, "persist"):
            memory.persist()
        return {"success": True, "message": f"Rewrote {params.label}"}
    except ValueError as e:
        return {"success": False, "error": str(e)}


@Registry.tool(
    name="core_memory_rethink",
    description="Compatibility alias for memory_rethink to rewrite a full core memory block.",
)
def core_memory_rethink_tool(params: MemoryRethinkParams, context: Dict[str, Any]):
    """Backwards-compatible alias for full core memory rewrites."""
    return memory_rethink_tool(params, context)


def register_core_tools(registry=None):
    """Tools auto-register on import; kept for compatibility."""
    return None
