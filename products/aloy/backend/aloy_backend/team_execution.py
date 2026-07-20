"""Building pori Teams from stored ``TeamConfig`` rows.

Shared by the HTTP layer (inline team execution in the conversations
messaging route) and the durable worker (``background.py``) — one
construction path so the two execution modes can't drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pori import (
    LLMConfig,
    MemberConfig,
    RunContext,
    Team,
    TeamMode,
    get_configured_llm,
    register_all_tools,
    tool_registry,
)

from .models import TeamConfig

if TYPE_CHECKING:
    from pori import AgentMemory, BudgetLedger


def build_team_from_config(
    team_config: TeamConfig,
    task: str,
    memory: "AgentMemory | None" = None,
    run_context: RunContext | None = None,
    budget_ledger: "BudgetLedger | None" = None,
) -> Team:
    """Construct a pori Team from a stored TeamConfig."""
    llm, _ = get_configured_llm()

    registry = tool_registry()
    register_all_tools(registry)

    members = []
    for m in team_config.members:
        llm_config = None
        if m.get("llm_config"):
            llm_config = LLMConfig(**m["llm_config"])

        members.append(
            MemberConfig(
                name=m["name"],
                description=m["description"],
                llm_config=llm_config,
                agent_settings=m.get("agent_settings"),
                tools=m.get("tools"),
            )
        )

    return Team(
        task=task,
        coordinator_llm=llm,
        members=members,
        mode=TeamMode(team_config.mode),
        tools_registry=registry,
        memory=memory,
        max_delegation_steps=team_config.max_delegation_steps,
        max_concurrent_members=team_config.max_concurrent_members,
        name=team_config.name,
        run_context=run_context,
        budget_ledger=budget_ledger,
    )
