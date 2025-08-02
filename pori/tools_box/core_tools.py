"""
Core tools required for agent operation.
"""

import logging
from pydantic import BaseModel, Field
from typing import Dict, Any


class AnswerParams(BaseModel):
    final_answer: str = Field(
        ..., description="The final answer to the user's question"
    )
    reasoning: str = Field(
        ..., description="Brief explanation of how you arrived at this answer"
    )


def answer_tool(params: AnswerParams, context: Dict[str, Any]) -> Dict[str, Any]:
    """Provide a final answer to the user's question."""
    answer = {
        "final_answer": params.final_answer,
        "reasoning": params.reasoning or "No additional reasoning provided.",
    }

    logging.info(f"Answer tool called with final answer: {params.final_answer}")

    # If context has memory, store this as the final answer
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


def done_tool(params: DoneParams, context: Dict[str, Any]):
    """Mark the task as done."""
    return {"final_message": params.message, "success": params.success}


class ThinkParams(BaseModel):
    thoughts: str = Field(..., description="What you're thinking about")
    next_action: str = Field(..., description="What you plan to do next")


def think_tool(params: ThinkParams, context: Dict[str, Any]):
    """Allow the agent to record its thoughts and planning."""
    return {
        "thoughts": params.thoughts,
        "next_action": params.next_action,
        "timestamp": "thinking step recorded",
    }


def register_core_tools(registry):
    """Register core tools required for agent operation."""
    registry.register_tool(
        name="answer",
        param_model=AnswerParams,
        function=answer_tool,
        description="REQUIRED: Provide your final answer to the user's question with reasoning",
    )

    registry.register_tool(
        name="done",
        param_model=DoneParams,
        function=done_tool,
        description="Mark the task as complete",
    )

    registry.register_tool(
        name="think",
        param_model=ThinkParams,
        function=think_tool,
        description="Record thoughts and planning (optional)",
    )
