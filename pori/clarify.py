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

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

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


# --- Gateway (buttons) --------------------------------------------------------


@dataclass(frozen=True)
class ClarificationRequest:
    """A structured clarification the client renders — options as tappable buttons
    plus a free-text field. This is the on-the-wire shape a gateway emits."""

    id: str
    question: str
    options: Tuple[str, ...] = ()

    def to_event(self) -> Dict[str, Any]:
        return {
            "type": "clarification_request",
            "id": self.id,
            "question": self.question,
            "options": list(self.options),
        }


class ClarifyBridge:
    """Gateway-side clarify handler that renders options as buttons.

    Transport-agnostic and reused by any gateway. ``emit(request)`` sends a
    :class:`ClarificationRequest` to the client (an SSE frame / websocket message
    the frontend renders as buttons); when the user answers, the transport's resume
    path calls :meth:`submit_answer`. The two calls are all a gateway needs — this
    class owns the pause/resume plumbing (an ``asyncio.Future`` per request).

    Wiring (no change to ``ask_user`` or the executor — the decoupling already
    exists): the gateway runs each agent turn in a worker thread and injects
    ``bridge.as_sync_handler(loop)`` as ``context["clarify_handler"]``; its resume
    endpoint calls ``bridge.submit_answer(id, value)``.
    """

    def __init__(
        self,
        emit: Callable[["ClarificationRequest"], Any],
        *,
        id_factory: Optional[Callable[[], str]] = None,
    ):
        self._emit = emit
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self._pending: Dict[str, "asyncio.Future[str]"] = {}

    async def ask(self, question: str, options: List[str]) -> str:
        """Emit a clarification and await the user's answer (an async handler)."""
        request = ClarificationRequest(
            id=self._id_factory(), question=question, options=tuple(options)
        )
        future: "asyncio.Future[str]" = asyncio.get_running_loop().create_future()
        self._pending[request.id] = future
        self._emit(request)
        try:
            return await future
        finally:
            self._pending.pop(request.id, None)

    def submit_answer(self, clarification_id: str, value: str) -> bool:
        """Resolve a pending clarification with the user's answer (button or text).

        Returns False if the id is unknown or already answered, so it's idempotent
        and safe against retries/double-taps. Safe to call from another thread.
        """
        future = self._pending.get(clarification_id)
        if future is None or future.done():
            return False
        future.get_loop().call_soon_threadsafe(future.set_result, value)
        return True

    def as_sync_handler(
        self, loop: "asyncio.AbstractEventLoop"
    ) -> Callable[[str, List[str]], str]:
        """A sync ``ClarifyHandler`` for runs executing OFF ``loop`` (a worker
        thread): schedules :meth:`ask` on ``loop`` and blocks the worker until the
        answer arrives. Do not use it for a run on ``loop`` itself — it would
        deadlock; use :meth:`ask` directly there (with async tool execution)."""

        def handler(question: str, options: List[str]) -> str:
            return asyncio.run_coroutine_threadsafe(
                self.ask(question, options), loop
            ).result()

        return handler

    def pending_ids(self) -> List[str]:
        return list(self._pending)
