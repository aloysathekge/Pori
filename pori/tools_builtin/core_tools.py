"""
Core tools required for agent operation.
"""

import logging
from pydantic import BaseModel, Field
from typing import Dict, Any
from ..tools import tool_registry

Registry = tool_registry()


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

    logging.info(f"Answer tool called with final answer: {params.final_answer}")

    if context and "memory" in context:
        context["memory"].update_state("final_answer", answer)
        logging.info("Final answer stored in memory")
    else:
        logging.warning(
            "Could not store final answer - memory not available in context"
        )

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


def register_core_tools(registry=None):
    """Tools auto-register on import; kept for compatibility."""
    return None
