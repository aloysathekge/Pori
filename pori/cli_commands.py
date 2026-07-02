"""Central slash-command registry (CLI-1).

Single source of truth for the CLI's slash commands. Every consumer — the
``/help`` listing, the unknown-command hint, and (later) tab-completion — derives
its data from ``COMMAND_REGISTRY`` instead of re-declaring commands, so the help
text can never drift from what the dispatcher actually handles (the exact
stale-help bug this replaces). To add a command: add a ``CommandDef`` here and
wire its handler in ``main._handle_cli_command``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CommandDef:
    name: str  # canonical, no slash, e.g. "memory"
    description: str
    category: str = "General"
    aliases: Tuple[str, ...] = ()
    args_hint: str = ""


COMMAND_REGISTRY: List[CommandDef] = [
    # Session
    CommandDef(
        "new",
        "Start a fresh conversation (keeps durable memory)",
        "Session",
        aliases=("reset", "clear"),
    ),
    CommandDef(
        "memory",
        "Inspect or clear memory",
        "Session",
        args_hint="[list|clear [all|messages|experiences|tasks|archival]]",
    ),
    # Configuration
    CommandDef("model", "Switch the active LLM provider/model", "Configuration"),
    # Skills
    CommandDef(
        "skills",
        "List or manage skills",
        "Skills",
        args_hint="[query|install <id>|uninstall <id>|inspect <id>]",
    ),
    CommandDef(
        "skill", "Show details for one skill", "Skills", args_hint="<name-or-id> [file]"
    ),
    CommandDef("reload-skills", "Reload local skills and bundles from disk", "Skills"),
    CommandDef(
        "evolution",
        "Manage self-improvement proposals",
        "Skills",
        args_hint="[list|show|propose|eval|approve|reject|activate|rollback]",
    ),
    # Info / control
    CommandDef("help", "Show this list of commands", "Info", aliases=("commands",)),
    CommandDef("cancel", "Cancel a pending skill invocation", "Info"),
    CommandDef("exit", "Exit the REPL", "Exit", aliases=("quit",)),
]


_LOOKUP: Dict[str, CommandDef] = {}
for _cmd in COMMAND_REGISTRY:
    _LOOKUP[_cmd.name] = _cmd
    for _alias in _cmd.aliases:
        _LOOKUP[_alias] = _cmd


def resolve_command(name: str) -> Optional[CommandDef]:
    """Resolve a command name (with or without a leading slash) or alias."""
    return _LOOKUP.get(name.lstrip("/").lower())


def known_command_names() -> List[str]:
    """Every canonical name and alias, slash-prefixed (for hints/completion)."""
    return [f"/{name}" for name in _LOOKUP]


def command_help_lines() -> List[str]:
    """Render grouped help lines from the registry (the one source of truth)."""
    by_category: Dict[str, List[CommandDef]] = {}
    for cmd in COMMAND_REGISTRY:
        by_category.setdefault(cmd.category, []).append(cmd)
    lines: List[str] = []
    for category, cmds in by_category.items():
        lines.append(f"\n{category}:")
        for cmd in cmds:
            usage = f"/{cmd.name}" + (f" {cmd.args_hint}" if cmd.args_hint else "")
            alias_note = (
                f"  (aliases: {', '.join('/' + a for a in cmd.aliases)})"
                if cmd.aliases
                else ""
            )
            lines.append(f"  {usage:<52} {cmd.description}{alias_note}")
    return lines
