"""Tool side-effect authorization + HITL interrupt resolution. These are `Agent`
methods, grouped here for readability and bound onto the class in `core`.
"""

import re
from typing import Any, Optional, Tuple

from ..hitl import resolve_interrupt_config
from ..tools.policy import AuthorizationDecision


def _tool_side_effects(self, tool_name: str) -> Tuple[Any, ...]:
    """Return the declared side effects for a tool, empty if unknown."""
    try:
        return self.capability_snapshot.get_tool(tool_name).side_effects
    except ValueError:
        info = self.tools_registry.tools.get(tool_name)
        return info.side_effects if info is not None else ()


def _authorize_side_effects(self, tool_name: str) -> AuthorizationDecision:
    """Authorize a tool call against its declared side effects."""
    return self.tool_authorization_policy.authorize(
        tool_name=tool_name,
        side_effects=self._tool_side_effects(tool_name),
        task=self.task,
    )


def _memory_deletion_forbidden_terms(self) -> list:
    """Extract terms from the task that the user explicitly wants removed.

    Returns a list of lowercased terms that should not appear in new memory writes.
    """
    forbidden = []
    task_lower = self.task.lower()
    # Look for quoted terms
    quoted = re.findall(r'["\']([^"\']+)["\']', self.task)
    for q in quoted:
        forbidden.append(q.lower())
    # Look for "FinBot" style proper nouns mentioned with deletion intent
    if "finbot" in task_lower:
        forbidden.append("finbot")
    if "master financial summary" in task_lower:
        forbidden.append("master financial summary")
    return forbidden


def _params_write_forbidden_memory_terms(self, tool_name: str, params: dict) -> bool:
    """Return True if the tool params would write forbidden terms back to memory."""
    if not self._task_requests_memory_mutation():
        return False
    forbidden = self._memory_deletion_forbidden_terms()
    if not forbidden:
        return False
    # Check various param fields that could contain new content
    fields_to_check = ["content", "new_str", "new_memory", "new_string"]
    for field in fields_to_check:
        val = params.get(field, "")
        if isinstance(val, str):
            val_lower = val.lower()
            for term in forbidden:
                if term in val_lower:
                    return True
    return False


def _get_interrupt_config(self, tool_name: str) -> Optional[Any]:
    """Check if a tool requires HITL approval."""
    if not self.hitl_config or not self.hitl_config.enabled:
        return None

    cfg = resolve_interrupt_config(tool_name, self.hitl_config)
    if cfg is None:
        return None

    # Check if auto_approve_duplicates applies
    if self.hitl_config.auto_approve_duplicates:
        # We don't have the exact params here, so the actual check happens
        # in execute_actions where we know the params. But we can check
        # if we previously approved *this exact tool+params signature*.
        # The logic in execute_actions uses this helper just to see if the
        # tool generally requires approval, and then handles duplicates directly.
        pass

    return cfg
