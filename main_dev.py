"""
Development version of main.py for testing API integrations.
This file includes experimental tools that are not part of the core framework.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from tools import ToolRegistry
from agent import AgentSettings
from orchestrator import Orchestrator
from tools_box import register_all_tools
from dev_tools import register_dev_tools

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def main():
    # Set up the tool registry with both core and dev tools
    registry = ToolRegistry()

    # Register core framework tools
    register_all_tools(registry)

    # Register development/experimental tools
    register_dev_tools(registry)

    # Create LLM
    llm = ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # Create orchestrator
    orchestrator = Orchestrator(llm=llm, tools_registry=registry)

    # Test task with Spotify API
    task = "Search for popular songs by Taylor Swift and tell me about the most popular one"

    try:
        result = await orchestrator.execute_task(
            task=task,
            agent_settings=AgentSettings(max_steps=15),
        )

        print("\n=== Task Execution Summary ===")
        print(f"Task: {task}")
        print(f"Success: {result['success']}")
        print(f"Steps taken: {result['steps_taken']}")

        # Show the final answer
        agent = result.get("agent")
        if agent:
            final_answer = agent.memory.get_state("final_answer")
            if final_answer:
                print("\n📝 FINAL ANSWER:")
                print(f"  {final_answer['final_answer']}")
                if final_answer.get("reasoning"):
                    print(f"\n  Reasoning: {final_answer['reasoning']}")
            else:
                print("\n⚠️ NO FINAL ANSWER FOUND")

            print("\nTool Calls:")
            for i, tool_call in enumerate(agent.memory.tool_call_history):
                status = "✓" if tool_call.success else "✗"
                print(
                    f"{i+1}. {tool_call.tool_name}({tool_call.parameters}) → {status}"
                )
                if not tool_call.success and tool_call.result.get("error"):
                    print(f"   Error: {tool_call.result['error']}")

    except Exception as e:
        print(f"Error executing task: {e}")


if __name__ == "__main__":
    print("🧪 Running development version with API tools")
    print("📝 Remember to set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env")
    print("🗑️ Delete dev_tools/ directory when done experimenting\n")

    asyncio.run(main())
