"""Cross-step loop / no-progress guardrail for tool calls (AC-5).

Pori already dedupes identical tool calls *within* a step; this catches the
slower failure modes that waste whole runs (up to ``max_steps``):

- a tool that fails with the SAME ``(name, args)`` across steps, and
- an idempotent read that returns the SAME result across steps (no progress).

The controller is pure apart from its own counters, so it is easy to unit test.
It never blocks a call outright: it emits a ``warn`` (a recovery hint appended to
the tool output the model sees) and, past a hard threshold, a ``halt`` the
runtime turns into ending the run. So it only ever fires on a *detected* loop —
in line with the repo's "no costly verification gates" rule.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# Read-only tools whose repeated identical output means "no new information".
# Mutating tools are excluded — re-running them can legitimately return the same
# acknowledgement without being a stuck loop.
IDEMPOTENT_TOOL_NAMES = {
    "read_file",
    "search_files",
    "list_files",
    "web_search",
    "web_extract",
    "get_state",
    "vision_analyze",
}


@dataclass(frozen=True)
class GuardrailDecision:
    action: str  # "warn" | "halt"
    reason: str
    guidance: str


@dataclass
class ToolCallGuardrailController:
    """Detect cross-step tool loops and return warn/halt decisions."""

    exact_failure_warn_after: int = 2
    exact_failure_halt_after: int = 3
    same_tool_halt_after: int = 6
    no_progress_warn_after: int = 3
    no_progress_halt_after: int = 5

    _exact_failures: Dict[str, int] = field(default_factory=dict)
    _same_tool_failures: Dict[str, int] = field(default_factory=dict)
    _last_result: Dict[str, Tuple[str, int]] = field(default_factory=dict)

    @staticmethod
    def signature(tool: str, params: Any) -> str:
        try:
            return f"{tool}:{json.dumps(params, sort_keys=True, default=str)}"
        except Exception:
            return f"{tool}:{params!r}"

    @staticmethod
    def _result_hash(result: Any) -> str:
        try:
            text = json.dumps(result, sort_keys=True, default=str)
        except Exception:
            text = repr(result)
        return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()

    def after_call(
        self, tool: str, params: Any, success: bool, result: Any
    ) -> Optional[GuardrailDecision]:
        """Observe a completed tool call and return a decision if a loop is seen."""
        sig = self.signature(tool, params)

        if not success:
            self._exact_failures[sig] = self._exact_failures.get(sig, 0) + 1
            self._same_tool_failures[tool] = self._same_tool_failures.get(tool, 0) + 1
            self._last_result.pop(sig, None)
            n = self._exact_failures[sig]
            if n >= self.exact_failure_halt_after:
                return GuardrailDecision(
                    "halt",
                    "exact_failure_loop",
                    f"'{tool}' has failed {n} times with identical arguments. Stop "
                    "retrying it unchanged — change the arguments or approach, or "
                    "finish with what you have.",
                )
            if self._same_tool_failures[tool] >= self.same_tool_halt_after:
                return GuardrailDecision(
                    "halt",
                    "same_tool_failure_loop",
                    f"'{tool}' has failed {self._same_tool_failures[tool]} times. Try a "
                    "different tool or approach, or finish with what you have.",
                )
            if n >= self.exact_failure_warn_after:
                return GuardrailDecision(
                    "warn",
                    "exact_failure",
                    f"'{tool}' has now failed {n} times with the same arguments. Change "
                    "the arguments or try another approach rather than repeating it.",
                )
            return None

        # Success clears the failure counters for this signature/tool.
        self._exact_failures.pop(sig, None)
        self._same_tool_failures.pop(tool, None)

        if tool in IDEMPOTENT_TOOL_NAMES:
            h = self._result_hash(result)
            prev_hash, count = self._last_result.get(sig, ("", 0))
            count = count + 1 if h == prev_hash else 1
            self._last_result[sig] = (h, count)
            if count >= self.no_progress_halt_after:
                return GuardrailDecision(
                    "halt",
                    "no_progress_loop",
                    f"'{tool}' has returned the same result {count} times. This is not "
                    "making progress — use the information you have and finish or move on.",
                )
            if count >= self.no_progress_warn_after:
                return GuardrailDecision(
                    "warn",
                    "no_progress",
                    f"'{tool}' returned the same result {count} times; you already have "
                    "this information — proceed rather than repeating the call.",
                )
        return None

    def reset(self) -> None:
        self._exact_failures.clear()
        self._same_tool_failures.clear()
        self._last_result.clear()
