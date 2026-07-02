"""Interactive prompt with slash-command completion + history (CLI-2).

Uses ``prompt_toolkit`` when it is installed for tab-completion of slash commands
(derived from the CommandDef registry — CLI-1) and persistent history, and falls
back to plain ``input()`` otherwise. The dependency is optional (extra ``cli``),
so the CLI keeps working with zero behavior change until a user opts in with
``pip install pori[cli]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cli_commands import known_command_names

_UNSET = object()
_session: Any = _UNSET  # sentinel: session not yet built (build lazily, once)


def _build_session() -> Any:
    """Build a prompt_toolkit session, or return None if it isn't available."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import FileHistory
    except ImportError:
        return None

    history: Any = None
    try:
        history_path = Path.home() / ".pori" / "cli_history"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history = FileHistory(str(history_path))
    except OSError:
        history = None

    completer = WordCompleter(known_command_names(), ignore_case=True)
    return PromptSession(history=history, completer=completer)


async def read_user_input(prompt: str) -> str:
    """Read a line, with slash-command completion + history when available."""
    global _session
    if _session is _UNSET:
        _session = _build_session()
    if _session is not None:
        return await _session.prompt_async(prompt)
    return input(prompt)
