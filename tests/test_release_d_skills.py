import pytest
from pydantic import BaseModel

from pori import (
    Agent,
    AgentMemory,
    BudgetExceeded,
    BudgetLedger,
    CancellationToken,
    ExecutionBudget,
    SkillCatalog,
    SkillManifest,
)
from pori.llm import ensure_budgeted_chat_model
from pori.tools.registry import ToolRegistry
from pori.tools.standard import register_all_tools


def test_skill_catalog_loads_only_selected_eligible_skill(tool_registry):
    loaded = {"research": 0, "report": 0}
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="research-news",
            name="Research News",
            version="1",
            summary="Search current news and cite sources",
            tags=("news", "research"),
            required_tools=frozenset({"test_tool"}),
        ),
        lambda: loaded.__setitem__("research", loaded["research"] + 1)
        or "Use test_tool to gather current evidence before answering.",
    )
    catalog.register(
        SkillManifest(
            slug="write-report",
            name="Write Report",
            version="1",
            summary="Format a long executive report",
            tags=("writing",),
        ),
        lambda: loaded.__setitem__("report", loaded["report"] + 1)
        or "Format the answer as a report.",
    )

    selected = catalog.select("research latest news", tool_registry.snapshot())
    skills = catalog.load_selected(selected)

    assert [skill.manifest.slug for skill in skills] == ["research-news"]
    assert loaded == {"research": 1, "report": 0}


def test_selected_skill_enters_agent_prompt_without_loading_others(
    mock_llm, tool_registry
):
    loaded = {"selected": 0, "other": 0}
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="database-migration",
            name="Database Migration",
            version="1",
            summary="Plan database migration and rollback",
            tags=("database", "migration"),
        ),
        lambda: loaded.__setitem__("selected", loaded["selected"] + 1)
        or "Always identify rollback and verification steps.",
    )
    catalog.register(
        SkillManifest(
            slug="graphic-design",
            name="Graphic Design",
            version="1",
            summary="Create visual brand assets",
        ),
        lambda: loaded.__setitem__("other", loaded["other"] + 1)
        or "Use visual design language.",
    )

    agent = Agent(
        task="plan a database migration",
        llm=mock_llm,
        tools_registry=tool_registry,
        skill_catalog=catalog,
        selected_skill_ids=["database-migration@1"],
    )

    assert loaded == {"selected": 1, "other": 0}
    assert "Always identify rollback" in agent.system_message
    assert "visual design language" not in agent.system_message
    assert agent.result_summary()["selected_skills"] == ["database-migration@1"]


def test_agent_run_result_reports_selected_skills(mock_llm, tool_registry):
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="teach",
            name="Teach",
            version="1",
            summary="Teach the user a new skill or concept",
        ),
        "Teach the user interactively.",
    )

    agent = Agent(
        task="teach me multiplication",
        llm=mock_llm,
        tools_registry=tool_registry,
        skill_catalog=catalog,
        selected_skill_ids=["teach@1"],
    )

    assert agent.result_summary()["selected_skills"] == ["teach@1"]


def test_agent_exposes_skill_index_without_auto_loading(mock_llm):
    registry = ToolRegistry()
    register_all_tools(registry)
    loaded = {"teach": 0}
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="teach",
            name="Teach",
            version="1",
            summary="Teach the user a new skill or concept",
            tags=("learning",),
        ),
        lambda: loaded.__setitem__("teach", loaded["teach"] + 1)
        or "Full teach instructions should only load through skill_view.",
    )

    agent = Agent(
        task="Explain why saying I would have created a file is not creating one.",
        llm=mock_llm,
        tools_registry=registry,
        skill_catalog=catalog,
    )

    assert loaded == {"teach": 0}
    assert "teach@1: Teach the user a new skill or concept" in agent.system_message
    assert "skills_list and skill_view" in agent.system_message
    assert "Full teach instructions" not in agent.system_message
    assert agent.result_summary()["selected_skills"] == []


@pytest.mark.asyncio
async def test_explicit_skill_workflow_request_must_load_skill_before_answer(mock_llm):
    registry = ToolRegistry()
    register_all_tools(registry)
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="teach",
            name="Teach",
            version="1",
            summary="Teach the user a new skill or concept",
            tags=("learning",),
        ),
        "Teach step by step.",
    )
    memory = AgentMemory()
    agent = Agent(
        task="Teach me division step by step simply",
        llm=mock_llm,
        tools_registry=registry,
        memory=memory,
        skill_catalog=catalog,
    )

    await agent.execute_actions(
        [
            {
                "answer": {
                    "final_answer": "Division is sharing.",
                    "reasoning": "Direct answer.",
                }
            }
        ]
    )

    answer_calls = [tc for tc in memory.tool_call_history if tc.tool_name == "answer"]
    assert answer_calls[0].success is False
    assert "Load teach@1 with skill_view before answering" in str(
        answer_calls[0].result
    )


@pytest.mark.asyncio
async def test_skill_with_model_invocation_disabled_is_not_nudged(mock_llm):
    """`disable-model-invocation` skills are user-invoked only, never auto-loaded."""
    registry = ToolRegistry()
    register_all_tools(registry)
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="teach",
            name="Teach",
            version="1",
            summary="Teach the user a new skill or concept",
            tags=("learning",),
            model_invocation_disabled=True,
        ),
        "Teach step by step.",
    )
    memory = AgentMemory()
    agent = Agent(
        task="Teach me division step by step simply",
        llm=mock_llm,
        tools_registry=registry,
        memory=memory,
        skill_catalog=catalog,
    )

    # Not advertised to the model, and not forced before answering.
    assert "teach@1" not in agent.system_message
    assert agent._required_skill_view_before_answer() is None

    await agent.execute_actions(
        [{"answer": {"final_answer": "Division is sharing.", "reasoning": "Direct."}}]
    )
    answer_calls = [tc for tc in memory.tool_call_history if tc.tool_name == "answer"]
    assert answer_calls[0].success is True


@pytest.mark.asyncio
async def test_explicit_skill_workflow_answer_allowed_after_skill_view(mock_llm):
    registry = ToolRegistry()
    register_all_tools(registry)
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="teach",
            name="Teach",
            version="1",
            summary="Teach the user a new skill or concept",
            tags=("learning",),
        ),
        "Teach step by step.",
    )
    memory = AgentMemory()
    agent = Agent(
        task="Teach me division step by step simply",
        llm=mock_llm,
        tools_registry=registry,
        memory=memory,
        skill_catalog=catalog,
    )

    await agent.execute_actions([{"skill_view": {"skill": "teach@1"}}])
    await agent.execute_actions(
        [
            {
                "answer": {
                    "final_answer": "Division is sharing.",
                    "reasoning": "Teach skill was loaded first.",
                }
            }
        ]
    )

    answer_calls = [tc for tc in memory.tool_call_history if tc.tool_name == "answer"]
    assert answer_calls[-1].success is True


def test_memory_context_is_fenced_below_current_task(mock_llm, tool_registry):
    memory = AgentMemory()
    memory.core_memory.update_block_value(
        "human",
        "The user's favourite cricketer is Quinton de Kock.",
    )
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="teach",
            name="Teach",
            version="1",
            summary="Teach the user a new skill or concept",
        ),
        "Teach the requested topic.",
    )

    agent = Agent(
        task="teach me something about science",
        llm=mock_llm,
        tools_registry=tool_registry,
        memory=memory,
        skill_catalog=catalog,
    )
    messages = agent._build_messages()

    assert "Quinton de Kock" not in messages[0].content
    assert any("<memory-context>" in message.content for message in messages)
    assert messages[-1].content.startswith("CURRENT TASK (highest priority):")
    assert "teach me something about science" in messages[-1].content


def test_skill_invocation_reports_missing_argument():
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="workshop",
            name="Workshop",
            version="1",
            summary="Run a structured workshop",
            commands=("workshop",),
            argument_hint="What topic should the workshop cover?",
        ),
        "Run the workshop.",
    )

    vague = catalog.build_invocation(
        "workshop@1",
        "please workshop something",
    )
    specific = catalog.build_invocation(
        "workshop@1",
        "please workshop API design",
    )

    assert vague.missing_argument is True
    assert vague.argument_hint == "What topic should the workshop cover?"
    assert specific.missing_argument is False
    assert specific.invocation_text == "api design"


def test_ineligible_selected_skill_is_rejected(tool_registry):
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="web-research",
            name="Web Research",
            version="1",
            summary="Search the web",
            required_tools=frozenset({"missing_web_tool"}),
        ),
        "Search the web first.",
    )

    with pytest.raises(ValueError, match="missing_tool"):
        catalog.select(
            "web research",
            tool_registry.snapshot(),
            explicit_skill_ids=["web-research@1"],
        )


def test_budget_ledger_enforces_shared_limits():
    ledger = BudgetLedger(
        ExecutionBudget(
            max_steps=2,
            max_tool_calls=2,
            max_tokens=10,
            max_cost_usd=1,
        )
    )

    ledger.consume_step()
    ledger.consume_step()
    with pytest.raises(BudgetExceeded):
        ledger.consume_step()

    ledger.consume_tool_call()
    ledger.consume_tool_call()
    with pytest.raises(BudgetExceeded, match="Tool-call") as exhausted:
        ledger.consume_tool_call()
    assert exhausted.value.code == "max_tool_calls"

    ledger.consume_tokens(10)
    with pytest.raises(BudgetExceeded):
        ledger.consume_tokens(1)

    ledger.consume_cost(1)
    with pytest.raises(BudgetExceeded):
        ledger.consume_cost(0.01)

    snapshot = ledger.snapshot()
    assert snapshot["tokens_used"] == 11
    assert snapshot["cost_used_usd"] == pytest.approx(1.01)


def test_budget_ledger_restores_usage_across_attempts():
    ledger = BudgetLedger(
        ExecutionBudget(max_steps=4, max_tool_calls=3, max_tokens=20),
        initial_usage={
            "steps_used": 2,
            "tool_calls_used": 3,
            "tokens_used": 12,
            "duration_seconds_used": 8.5,
        },
    )

    ledger.consume_step()
    with pytest.raises(BudgetExceeded) as exhausted:
        ledger.consume_tool_call()

    assert exhausted.value.code == "max_tool_calls"
    snapshot = ledger.snapshot()
    assert snapshot["steps_used"] == 3
    assert snapshot["tool_calls_used"] == 3
    assert snapshot["tokens_used"] == 12
    assert snapshot["duration_seconds_used"] == pytest.approx(8.5)


@pytest.mark.asyncio
async def test_budgeted_model_charges_structured_and_ordinary_calls():
    class Payload(BaseModel):
        value: str

    class MeteredModel:
        model = "gpt-4o-mini"

        def __init__(self):
            self.last_usage = None

        def with_structured_output(self, output_model, include_raw=False):
            return self

        async def ainvoke(self, messages, output_format=None):
            self.last_usage = {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            }
            return Payload(value="ok")

    ledger = BudgetLedger(ExecutionBudget(max_tokens=100, max_cost_usd=1))
    model = ensure_budgeted_chat_model(MeteredModel(), ledger)

    await model.with_structured_output(Payload).ainvoke([])
    await model.ainvoke([])

    snapshot = ledger.snapshot()
    assert snapshot["llm_calls_used"] == 2
    assert snapshot["input_tokens_used"] == 20
    assert snapshot["output_tokens_used"] == 10
    assert snapshot["tokens_used"] == 30
    assert snapshot["cost_used_usd"] > 0
    assert snapshot["unpriced_llm_calls"] == 0


@pytest.mark.asyncio
async def test_cancellation_token_stops_agent_before_first_step(
    mock_llm, tool_registry
):
    token = CancellationToken()
    token.cancel()
    agent = Agent(
        task="answer directly",
        llm=mock_llm,
        tools_registry=tool_registry,
        cancellation_token=token,
    )

    result = await agent.run()

    assert result["steps_taken"] == 0
    assert result["completed"] is False
