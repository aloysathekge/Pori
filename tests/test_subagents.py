"""Sub-agent task delegation: definitions, catalog, runner, tool, isolated run."""

import pytest

from pori.orchestrator.core import Orchestrator
from pori.subagents import (
    GENERAL_PURPOSE,
    AgentCatalog,
    make_subagent_runner,
    parse_agent_markdown,
)
from pori.tools.standard.core_tools import TaskParams, task_tool

pytestmark = [pytest.mark.unit]


# --- definitions + catalog --------------------------------------------------
def test_parse_agent_markdown_full():
    md = (
        "---\nname: code-explorer\ndescription: Explore code paths.\n"
        "tools: Read, Grep\nmodel: sonnet\n---\nYou are an explorer."
    )
    d = parse_agent_markdown(md, fallback_name="fb")
    assert d.name == "code-explorer"
    assert d.description == "Explore code paths."
    assert d.tools == ("Read", "Grep")
    assert d.model == "sonnet"
    assert d.prompt == "You are an explorer."


@pytest.mark.parametrize(
    "text",
    [
        "no frontmatter here",
        "---\nname: valid-name\n---\nbody but no description",  # missing description
        "---\ndescription: d\nname: valid\n---\n",  # missing body
    ],
)
def test_parse_agent_markdown_invalid(text):
    assert parse_agent_markdown(text, fallback_name="fb") is None


def test_catalog_loads_dir_and_always_has_general_purpose(tmp_path):
    (tmp_path / "explorer.md").write_text(
        "---\nname: explorer\ndescription: Explore.\n---\nYou explore.",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    catalog = AgentCatalog.load(tmp_path)
    assert catalog.resolve("explorer").description == "Explore."
    assert catalog.resolve(GENERAL_PURPOSE) is not None  # built-in default
    assert catalog.resolve("missing") is None
    assert "explorer" in catalog.describe_types()


# --- runner -----------------------------------------------------------------
def test_runner_resolves_and_runs():
    class _StubOrch:
        async def run_subagent(
            self, task, *, system_prompt=None, tool_names=None, max_steps=15
        ):
            return f"ran:{task}|tools={tool_names}"

    runner = make_subagent_runner(_StubOrch(), AgentCatalog())
    assert runner(GENERAL_PURPOSE, "do X") == "ran:do X|tools=None"


def test_runner_unknown_type_raises_with_available():
    runner = make_subagent_runner(object(), AgentCatalog())
    with pytest.raises(ValueError) as exc:
        runner("nope", "x")
    assert GENERAL_PURPOSE in str(exc.value)


# --- the task tool ----------------------------------------------------------
def test_task_tool_delegates_via_runner():
    res = task_tool(
        TaskParams(subagent_type=GENERAL_PURPOSE, task="do X"),
        {"subagent_runner": lambda st, t: f"result:{t}"},
    )
    assert res["success"] is True and res["result"] == "result:do X"


def test_task_tool_without_runner_refuses_nesting():
    res = task_tool(TaskParams(subagent_type=GENERAL_PURPOSE, task="x"), {})
    assert res["success"] is False and "not available" in res["error"].lower()


def test_task_tool_surfaces_runner_error():
    def boom(subagent_type, task):
        raise ValueError("bad subagent")

    res = task_tool(TaskParams(subagent_type="x", task="y"), {"subagent_runner": boom})
    assert res["success"] is False and "bad subagent" in res["error"]


# --- the real isolated run --------------------------------------------------
async def test_run_subagent_returns_answer_with_isolated_memory(
    mock_llm, tool_registry
):
    orch = Orchestrator(llm=mock_llm, tools_registry=tool_registry)
    answer = await orch.run_subagent("say hi", system_prompt="be terse", max_steps=3)
    assert isinstance(answer, str) and answer
    # isolation: the sub-agent used its own memory, so the orchestrator's shared
    # memory was never populated by it.
    assert orch.shared_memory is None
