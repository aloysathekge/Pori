"""Live demo of the Team system with a real LLM.

Usage:
    uv run python examples/team_demo.py <mode> "<task>"

Examples:
    uv run python examples/team_demo.py router "What is 25 * 47?"
    uv run python examples/team_demo.py broadcast "Should I use Redis or Memcached for caching?"
    uv run python examples/team_demo.py delegate "Find the population of Tokyo, then compare it to New York"
"""

import asyncio
import sys

from pori import Team, TeamMode, MemberConfig
from pori.agent import AgentSettings
from pori.config import load_config, create_llm
from pori.tools.registry import tool_registry


MEMBER_SETTINGS = AgentSettings(max_steps=5, max_failures=2)

# -- Member presets per mode --

ROUTER_MEMBERS = [
    MemberConfig(name="general", description="Answers general knowledge and trivia questions"),
    MemberConfig(name="math", description="Solves math, arithmetic, and numerical problems"),
    MemberConfig(name="coder", description="Writes code, explains programming concepts, debugs"),
]

BROADCAST_MEMBERS = [
    MemberConfig(name="optimist", description="Focuses on benefits, opportunities, and upsides"),
    MemberConfig(name="skeptic", description="Focuses on risks, drawbacks, and trade-offs"),
    MemberConfig(name="pragmatist", description="Focuses on practical, real-world considerations"),
]

DELEGATE_MEMBERS = [
    MemberConfig(name="researcher", description="Gathers information, looks up facts, collects data"),
    MemberConfig(name="analyst", description="Analyses data, compares options, draws conclusions"),
    MemberConfig(name="writer", description="Writes clear summaries, reports, and explanations"),
]


async def run_team(mode: str, task: str):
    config = load_config()
    llm = create_llm(config.llm)
    registry = tool_registry()

    mode_enum = TeamMode(mode)

    members_map = {
        TeamMode.ROUTER: ROUTER_MEMBERS,
        TeamMode.BROADCAST: BROADCAST_MEMBERS,
        TeamMode.DELEGATE: DELEGATE_MEMBERS,
    }

    team = Team(
        task=task,
        coordinator_llm=llm,
        members=members_map[mode_enum],
        mode=mode_enum,
        tools_registry=registry,
        agent_defaults=MEMBER_SETTINGS,
        name=f"{mode}-demo",
    )

    result = await team.run()

    print(f"\n{'='*60}")
    print(f"  RESULT ({mode.upper()} MODE)")
    print(f"{'='*60}")
    print(f"  Completed : {result['completed']}")
    print(f"  Steps     : {result['steps_taken']}")

    fs = result.get("final_state", {})
    if "chosen_member" in fs:
        print(f"  Routed to : {fs['chosen_member']}")
    if "plan_steps" in fs:
        print(f"  Plan steps  : {fs['plan_steps']} (delegation steps)")
        print(f"  Agent steps : {fs['agent_steps']} (total across all member agents)")
    print(f"\n  Answer:\n  {fs.get('final_answer', '(none)')}")


if __name__ == "__main__":
    modes = ("router", "broadcast", "delegate")

    if len(sys.argv) < 3 or sys.argv[1] not in modes:
        print(f"Usage: uv run python examples/team_demo.py <{'|'.join(modes)}> \"<task>\"")
        print()
        print("Examples:")
        print('  uv run python examples/team_demo.py router "What is 25 * 47?"')
        print('  uv run python examples/team_demo.py broadcast "Should I use Redis or Memcached for caching?"')
        print('  uv run python examples/team_demo.py delegate "Find the population of Tokyo, then compare it to New York"')
        sys.exit(1)

    mode = sys.argv[1]
    task = sys.argv[2]

    print(f"Running {mode} mode...")
    asyncio.run(run_team(mode, task))
