"""Structured user clarification, decoupled from transport.

The ``ask_user`` tool asks the user a question, optionally offering a fixed set of
choices. *How* that question is presented and answered is a transport concern: the
CLI renders a numbered menu on stdin; a gateway/API can render tappable buttons.

The tool calls a ``ClarifyHandler`` supplied in the tool context if one is present,
and otherwise falls back to the built-in CLI menu — the same decoupling Pori uses
for HITL handlers, so the agent's behavior is identical everywhere and only the
presentation changes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # Protocol is nice-to-have; fall back cleanly on very old runtimes.
    from typing import Protocol

    class ClarifyHandler(Protocol):
        def __call__(self, question: str, options: List[str]) -> str: ...

except ImportError:  # pragma: no cover
    ClarifyHandler = Any  # type: ignore


def _read_line(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def cli_clarify(question: str, options: List[str]) -> str:
    """Default CLI clarification: a numbered menu, or free text when no options.

    Returns the chosen option's text, or the user's free-text answer. Typing the
    "Other" number (or anything non-numeric) yields a free-text answer, so the
    user is never boxed in by the offered choices.
    """
    print(f"\n[Agent needs clarification] {question}")
    if not options:
        return _read_line("Your answer: ")

    for index, option in enumerate(options, 1):
        print(f"  {index}) {option}")
    other = len(options) + 1
    print(f"  {other}) Other (type your own)")

    raw = _read_line("Choose a number, or type your own answer: ")
    if raw.isdigit():
        choice = int(raw)
        if 1 <= choice <= len(options):
            return options[choice - 1]
        if choice == other:
            return _read_line("Your answer: ")
    return raw  # non-numeric input is taken as a free-text answer


def resolve_clarify_handler(context: Optional[Dict[str, Any]]) -> "ClarifyHandler":
    """Return the context's clarify handler if callable, else the CLI default."""
    handler = (context or {}).get("clarify_handler")
    return handler if callable(handler) else cli_clarify
