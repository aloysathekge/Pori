"""Declarative authorization for side-effecting tool calls.

Keeps the agent loop free of hardcoded tool-name sets and inline keyword
heuristics. Tools advertise their side effects through the registry
(:class:`~pori.tools.registry.SideEffect`); a single :class:`ToolAuthorizationPolicy`
decides whether a side-effecting call is authorized for the current task.

The intent heuristic lives here, isolated and testable, rather than embedded in
the agent. Swap or disable it without touching the execution loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .registry import SideEffect

# Terms that signal the user wants a durable artifact produced on disk.
_ARTIFACT_TERMS = (
    "file",
    "html",
    "page",
    "document",
    "report",
    "worksheet",
    "artifact",
    "notes/",
    ".html",
    ".md",
    ".txt",
    ".json",
)
# Terms that signal an explicit request to produce/persist something.
_WRITE_TERMS = (
    "create",
    "write",
    "save",
    "build",
    "generate",
    "make",
    "export",
    "put it in",
)


def task_requests_artifact(task: str) -> bool:
    """Return True when the task explicitly asks for a created file artifact."""
    task_lower = task.casefold()
    return any(term in task_lower for term in _ARTIFACT_TERMS) and any(
        term in task_lower for term in _WRITE_TERMS
    )


@dataclass(frozen=True)
class AuthorizationDecision:
    """The outcome of authorizing a single tool call."""

    allowed: bool
    reason: str = ""


class ToolAuthorizationPolicy:
    """Authorize side-effecting tool calls.

    By default the model decides whether a filesystem write is wanted: a request
    like "create me a python script" should write a file, and truthfulness is
    guaranteed separately by artifact receipts, not by guessing intent from
    keywords. Hard "ask before writing" gating belongs in the HITL layer.

    Set ``require_artifact_intent=True`` for a strict mode (e.g. locked-down
    sandboxes) that blocks filesystem writes unless the task explicitly asked
    for a created/written artifact.
    """

    def __init__(self, *, require_artifact_intent: bool = False):
        self.require_artifact_intent = require_artifact_intent

    def authorize(
        self,
        *,
        tool_name: str,
        side_effects: Iterable[SideEffect],
        task: str,
    ) -> AuthorizationDecision:
        effects = frozenset(side_effects)
        if (
            self.require_artifact_intent
            and SideEffect.FILESYSTEM_WRITE in effects
            and not task_requests_artifact(task)
        ):
            return AuthorizationDecision(
                allowed=False,
                reason=(
                    f"Tool '{tool_name}' would create or modify a filesystem "
                    "artifact, but the current task did not explicitly ask for a "
                    "file artifact. Answer in chat or ask the user before writing "
                    "files or directories."
                ),
            )
        return AuthorizationDecision(allowed=True)


__all__ = [
    "AuthorizationDecision",
    "ToolAuthorizationPolicy",
    "task_requests_artifact",
]
