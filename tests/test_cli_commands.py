"""Central slash-command registry (CLI-1)."""

import pytest

from pori.cli_commands import (
    COMMAND_REGISTRY,
    command_help_lines,
    known_command_names,
    resolve_command,
)

pytestmark = [pytest.mark.unit]


def test_resolve_handles_slash_and_aliases():
    assert resolve_command("/memory").name == "memory"
    assert resolve_command("memory").name == "memory"
    assert resolve_command("/reset").name == "new"  # alias -> canonical
    assert resolve_command("quit").name == "exit"
    assert resolve_command("/nope") is None


def test_help_lines_cover_every_command():
    text = "\n".join(command_help_lines())
    for cmd in COMMAND_REGISTRY:
        assert f"/{cmd.name}" in text
        assert cmd.description in text


def test_no_duplicate_names_or_aliases():
    seen = set()
    for cmd in COMMAND_REGISTRY:
        for key in (cmd.name, *cmd.aliases):
            assert key not in seen, f"duplicate command key: {key}"
            seen.add(key)


def test_known_names_are_slash_prefixed():
    names = known_command_names()
    assert "/memory" in names and "/reset" in names
    assert all(n.startswith("/") for n in names)


def test_dispatched_commands_are_registered():
    # Contract: every command main._handle_cli_command dispatches lives in the
    # registry, so /help and the unknown-command hint can't drift from dispatch.
    for name in (
        "new",
        "memory",
        "model",
        "skills",
        "skill",
        "reload-skills",
        "evolution",
        "help",
    ):
        assert resolve_command(name) is not None, name
