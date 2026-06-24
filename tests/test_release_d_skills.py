import pytest

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
    )

    assert agent.result_summary()["selected_skills"] == ["teach@1"]


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
    ledger = BudgetLedger(ExecutionBudget(max_steps=2, max_tokens=10, max_cost_usd=1))

    ledger.consume_step()
    ledger.consume_step()
    with pytest.raises(BudgetExceeded):
        ledger.consume_step()

    ledger.consume_tokens(10)
    with pytest.raises(BudgetExceeded):
        ledger.consume_tokens(1)

    ledger.consume_cost(1)
    with pytest.raises(BudgetExceeded):
        ledger.consume_cost(0.01)


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
