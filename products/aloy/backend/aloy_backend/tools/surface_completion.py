"""Host-enforced completion gate for purpose-scoped Surface Builder Runs."""

from __future__ import annotations

from typing import Any

from pori.tools.standard.core_tools import AnswerParams, answer_tool

from ..surface_requests import SurfaceBuilderCompletionGuard

SURFACE_COMPLETION_CONTEXT_KEY = "surface_completion_guard"


async def surface_builder_answer_tool(
    params: AnswerParams,
    context: dict,
) -> dict[str, Any]:
    guard = context.get(SURFACE_COMPLETION_CONTEXT_KEY)
    if not isinstance(guard, SurfaceBuilderCompletionGuard):
        raise ValueError("Surface Builder completion verification is unavailable")
    await guard.require_publication()
    return answer_tool(params, context)


def register_surface_builder_completion_tool(registry) -> None:
    registry.register_tool(
        name="answer",
        param_model=AnswerParams,
        function=surface_builder_answer_tool,
        description=(
            "Provide the final Surface Builder answer only after this exact Run "
            "has published a verified live Surface. Calling this earlier fails "
            "and the Run must continue authoring, building, previewing, and publishing."
        ),
    )


__all__ = [
    "SURFACE_COMPLETION_CONTEXT_KEY",
    "register_surface_builder_completion_tool",
    "surface_builder_answer_tool",
]
