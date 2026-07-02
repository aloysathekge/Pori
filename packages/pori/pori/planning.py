"""Model-owned plan/todo state for a single agent run.

Planning is model-driven (see `.agent/reference-studies/planning-architecture.md`):
the agent maintains a short todo list by calling the `update_plan` tool, rather
than the framework making a separate planning LLM call. The list is advisory and
run-scoped — it is NOT written to long-term memory.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from pydantic import BaseModel, ConfigDict

PLAN_STATUSES: Tuple[str, ...] = ("pending", "in_progress", "completed", "cancelled")
ACTIVE_STATUSES: frozenset[str] = frozenset({"pending", "in_progress"})
MAX_PLAN_ITEMS = 64
MAX_PLAN_CONTENT_CHARS = 2000

_STATUS_MARKS = {
    "pending": "[ ]",
    "in_progress": "[>]",
    "completed": "[x]",
    "cancelled": "[~]",
}


class PlanItem(BaseModel):
    """One todo item: an ordered, status-bearing step the model owns."""

    model_config = ConfigDict(frozen=True)

    id: str
    content: str
    status: str = "pending"


class PlanStore:
    """Ordered, model-owned todo list for one run (position = priority)."""

    def __init__(self) -> None:
        self._items: List[PlanItem] = []

    @staticmethod
    def _normalize(raw: Any, index: int) -> PlanItem | None:
        if isinstance(raw, dict):
            data = raw
        else:
            data = {"content": str(raw)}
        content = str(data.get("content") or "").strip()[:MAX_PLAN_CONTENT_CHARS]
        if not content:
            return None
        item_id = str(data.get("id") or "").strip() or str(index + 1)
        status = str(data.get("status") or "pending").strip().lower()
        if status not in PLAN_STATUSES:
            status = "pending"
        return PlanItem(id=item_id, content=content, status=status)

    def write(
        self, todos: Iterable[Any], *, merge: bool = False
    ) -> Tuple[PlanItem, ...]:
        """Replace the whole list (default) or update-by-id and append (merge)."""
        normalized = [
            item
            for index, raw in enumerate(todos)
            if (item := self._normalize(raw, index)) is not None
        ]
        if merge:
            by_id: Dict[str, PlanItem] = {item.id: item for item in self._items}
            order: List[str] = [item.id for item in self._items]
            for item in normalized:
                if item.id not in by_id:
                    order.append(item.id)
                by_id[item.id] = item
            ordered = [by_id[item_id] for item_id in order]
        else:
            # Replace; de-duplicate by id keeping last, preserving first-seen order.
            by_id = {}
            order = []
            for item in normalized:
                if item.id not in by_id:
                    order.append(item.id)
                by_id[item.id] = item
            ordered = [by_id[item_id] for item_id in order]
        self._items = ordered[:MAX_PLAN_ITEMS]
        return tuple(self._items)

    def items(self) -> Tuple[PlanItem, ...]:
        return tuple(self._items)

    def active(self) -> Tuple[PlanItem, ...]:
        return tuple(item for item in self._items if item.status in ACTIVE_STATUSES)

    def has_items(self) -> bool:
        return bool(self._items)

    def summary(self) -> Dict[str, int]:
        counts = {status: 0 for status in PLAN_STATUSES}
        for item in self._items:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    def format_for_prompt(self) -> str:
        """Render only active (pending/in_progress) items as a checklist.

        Completed/cancelled items are omitted so the model does not redo or dwell
        on finished work.
        """
        active = self.active()
        if not active:
            return ""
        return "\n".join(
            f"{_STATUS_MARKS.get(item.status, '[ ]')} {item.content}" for item in active
        )


__all__ = [
    "ACTIVE_STATUSES",
    "MAX_PLAN_CONTENT_CHARS",
    "MAX_PLAN_ITEMS",
    "PLAN_STATUSES",
    "PlanItem",
    "PlanStore",
]
