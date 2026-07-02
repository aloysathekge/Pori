"""Model-driven planning tool: the agent maintains its own todo list.

Behavioral guidance lives entirely in the tool schema description (cacheable),
mirroring the Hermes `todo` pattern. There is no separate planning LLM call.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ...planning import PlanStore
from ..registry import ToolRegistry

_UPDATE_PLAN_DESCRIPTION = (
    "Maintain a short task plan (todo list) for multi-step work. Call this for "
    "tasks needing 3+ distinct steps; skip it for trivial single-step tasks. "
    "Pass the full list of todos (ordered by priority). Rules: keep exactly ONE "
    "item 'in_progress' at a time; mark an item 'completed' as soon as it is done; "
    "if a step fails or becomes wrong, set it 'cancelled' and add a revised item. "
    "Do NOT add steps to gather information already present in your context or "
    "memory (e.g. do not plan to ask for a value you already have). Use merge=true "
    "to update items by id and append new ones; otherwise the list is replaced."
)


class PlanItemParam(BaseModel):
    id: str = Field(
        "",
        description="Stable id for the item; omit to auto-number by position.",
    )
    content: str = Field(..., description="Imperative step text, e.g. 'Run the tests'.")
    status: str = Field(
        "pending",
        description="One of: pending, in_progress, completed, cancelled.",
    )


class UpdatePlanParams(BaseModel):
    todos: List[PlanItemParam] = Field(
        default_factory=list,
        description="The full ordered todo list (or, with merge=true, items to upsert).",
    )
    merge: bool = Field(
        False,
        description="Update items by id and append new ones instead of replacing all.",
    )


def _plan_store_from_context(context: Dict[str, Any]) -> Optional[PlanStore]:
    store = context.get("plan_store")
    return store if isinstance(store, PlanStore) else None


def register_planning_tools(registry: ToolRegistry) -> None:
    """Register the model-driven planning tool on the provided registry."""

    @registry.tool(
        name="update_plan",
        param_model=UpdatePlanParams,
        description=_UPDATE_PLAN_DESCRIPTION,
    )
    def update_plan_tool(params: UpdatePlanParams, context: Dict[str, Any]):
        store = _plan_store_from_context(context)
        if store is None:
            return {
                "available": False,
                "error": "No plan store is configured for this run.",
            }
        items = store.write(
            [item.model_dump() for item in params.todos],
            merge=params.merge,
        )
        return {
            "available": True,
            "plan": [item.model_dump() for item in items],
            "summary": store.summary(),
        }


__all__ = ["PlanItemParam", "UpdatePlanParams", "register_planning_tools"]
