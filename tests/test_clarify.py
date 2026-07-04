"""Structured user clarification (ask_user with options)."""

import builtins

import pytest

from pori.clarify import cli_clarify, resolve_clarify_handler
from pori.tools.standard.core_tools import AskUserParams, ask_user_tool

pytestmark = [pytest.mark.unit]


def _inputs(monkeypatch, *values):
    stream = iter(values)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(stream))


def test_free_text_when_no_options(monkeypatch):
    _inputs(monkeypatch, "a red bicycle")
    assert cli_clarify("What?", []) == "a red bicycle"


def test_numeric_choice_returns_that_option(monkeypatch):
    _inputs(monkeypatch, "2")
    assert cli_clarify("Pick", ["Alpha", "Beta", "Gamma"]) == "Beta"


def test_other_choice_prompts_for_free_text(monkeypatch):
    # 2 options -> "Other" is choice 3, which asks again for free text.
    _inputs(monkeypatch, "3", "my own answer")
    assert cli_clarify("Pick", ["Alpha", "Beta"]) == "my own answer"


@pytest.mark.parametrize("typed", ["neither, actually", "9"])
def test_non_option_input_is_taken_as_free_text(monkeypatch, typed):
    _inputs(monkeypatch, typed)
    assert cli_clarify("Pick", ["Alpha", "Beta"]) == typed


def test_resolve_handler_prefers_context_then_falls_back():
    assert (
        resolve_clarify_handler({"clarify_handler": lambda q, o: "X"})("q", []) == "X"
    )
    assert resolve_clarify_handler({}) is cli_clarify
    assert resolve_clarify_handler(None) is cli_clarify


def test_ask_user_tool_routes_through_handler():
    seen = {}

    def handler(question, options):
        seen["question"] = question
        seen["options"] = options
        return "Beta"

    res = ask_user_tool(
        AskUserParams(question="Pick one", reason="blocked", options=["Alpha", "Beta"]),
        {"clarify_handler": handler},
    )
    assert res["success"] is True
    assert res["user_response"] == "Beta"
    assert seen["options"] == ["Alpha", "Beta"]


def test_agent_stores_tool_context_extra(mock_llm, tool_registry):
    from pori.agent import Agent
    from pori.memory import AgentMemory

    extra = {"clarify_handler": lambda q, o: "x"}
    agent = Agent(
        task="t",
        llm=mock_llm,
        tools_registry=tool_registry,
        tool_context_extra=extra,
        memory=AgentMemory(),
    )
    assert agent._tool_context_extra == extra
