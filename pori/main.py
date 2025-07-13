import asyncio
import logging
import os
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from tools import ToolRegistry, ToolExecutor
from agent import Agent, AgentSettings
from orchestrator import Orchestrator
from tools_box import register_all_tools

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Define some example tools


async def main():
    # Set up the tool registry
    registry = ToolRegistry()

    register_all_tools(registry)

    # Create LLM - uses ANTHROPIC_API_KEY from environment
    llm = ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # Create orchestrator
    orchestrator = Orchestrator(llm=llm, tools_registry=registry)

    # Define steps callback for monitoring
    def on_step_end(agent: Agent):
        print(f"Completed step {agent.state.n_steps}")

    # Execute a task
    print(" Pori Agent at your service!")
    task = input("How can I help you today? ")

    if not task.strip():
        print("No task provided. Exiting...")
        return

    try:
        result = await orchestrator.execute_task(
            task=task,
            agent_settings=AgentSettings(max_steps=10),
            on_step_end=on_step_end,
        )

        print("\n=== Task Execution Summary ===")
        print(f"Task: {task}")
        print(f"Success: {result['success']}")
        print(f"Steps taken: {result['steps_taken']}")

        # Show the final answer if available
        agent = result.get("agent")

        if agent:
            # Get final answer from memory
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
                print(
                    f"{i+1}. {tool_call.tool_name}({tool_call.parameters}) → {'✓' if tool_call.success else '✗'}"
                )

    except Exception as e:
        print(f"Error executing task: {e}")


if __name__ == "__main__":
    asyncio.run(main())
