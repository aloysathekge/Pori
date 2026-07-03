"""Sub-agent task delegation: definitions, catalog, runner, tool, isolated run."""

import pytest

from pori.orchestrator.core import Orchestrator
from pori.subagents import (
    GENERAL_PURPOSE,
    MAX_PARALLEL_SUBAGENTS,
    AgentCatalog,
    make_parallel_subagent_runner,
    make_subagent_runner,
    parse_agent_markdown,
)
from pori.tools.standard.core_tools import (
    ParallelTaskItem,
    TaskParallelParams,
    TaskParams,
    task_parallel_tool,
    task_tool,
)

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


# --- parallel delegation ----------------------------------------------------
def test_parallel_runner_collects_per_task_results_and_isolates_failures():
    class _StubOrch:
        async def run_subagent(
            self, task, *, system_prompt=None, tool_names=None, max_steps=15
        ):
            return f"done:{task}"

    runner = make_parallel_subagent_runner(_StubOrch(), AgentCatalog())
    results = runner([(GENERAL_PURPOSE, "a"), (GENERAL_PURPOSE, "b"), ("nope", "c")])
    assert len(results) == 3
    assert results[0]["success"] is True and results[0]["result"] == "done:a"
    assert results[1]["result"] == "done:b"
    assert results[2]["success"] is False and "Unknown" in results[2]["error"]


def test_parallel_runner_actually_runs_concurrently():
    import asyncio

    state = {"active": 0, "max": 0}

    class _SlowOrch:
        async def run_subagent(
            self, task, *, system_prompt=None, tool_names=None, max_steps=15
        ):
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
            await asyncio.sleep(0.05)
            state["active"] -= 1
            return task

    runner = make_parallel_subagent_runner(_SlowOrch(), AgentCatalog())
    runner([(GENERAL_PURPOSE, "a"), (GENERAL_PURPOSE, "b"), (GENERAL_PURPOSE, "c")])
    assert state["max"] >= 2  # they overlapped -> genuinely concurrent


def test_task_parallel_tool_delegates():
    def runner(items):
        return [{"subagent_type": st, "success": True, "result": t} for st, t in items]

    res = task_parallel_tool(
        TaskParallelParams(
            tasks=[
                ParallelTaskItem(subagent_type=GENERAL_PURPOSE, task="a"),
                ParallelTaskItem(subagent_type=GENERAL_PURPOSE, task="b"),
            ]
        ),
        {"parallel_subagent_runner": runner},
    )
    assert res["success"] is True and res["count"] == 2
    assert res["results"][0]["result"] == "a"


def test_task_parallel_without_runner_errors():
    res = task_parallel_tool(
        TaskParallelParams(
            tasks=[ParallelTaskItem(subagent_type=GENERAL_PURPOSE, task="a")]
        ),
        {},
    )
    assert res["success"] is False


def test_task_parallel_rejects_too_many():
    items = [
        ParallelTaskItem(subagent_type=GENERAL_PURPOSE, task=str(i))
        for i in range(MAX_PARALLEL_SUBAGENTS + 1)
    ]
    res = task_parallel_tool(
        TaskParallelParams(tasks=items), {"parallel_subagent_runner": lambda x: []}
    )
    assert res["success"] is False and "max" in res["error"].lower()
