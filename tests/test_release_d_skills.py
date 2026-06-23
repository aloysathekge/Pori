import pytest

from pori import (
    Agent,
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
