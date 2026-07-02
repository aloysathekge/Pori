"""Interactive prompt with graceful fallback (CLI-2)."""

import builtins

import pytest

import pori.cli_prompt as cli_prompt

pytestmark = [pytest.mark.unit]


async def test_read_user_input_falls_back_to_plain_input(monkeypatch):
    monkeypatch.setattr(cli_prompt, "_session", None)  # force the input() fallback
    monkeypatch.setattr(builtins, "input", lambda prompt="": "do the thing")
    assert (await cli_prompt.read_user_input("> ")) == "do the thing"


def test_build_session_degrades_to_none_without_prompt_toolkit(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("prompt_toolkit"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert cli_prompt._build_session() is None
