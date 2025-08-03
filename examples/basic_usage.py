"""
Basic usage example for Pori Agent Framework.

This example shows how to use the framework.
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from pori import Agent, AgentSettings, ToolRegistry, Orchestrator, register_all_tools

# Load environment variables
load_dotenv()


async def main():
    """Basic example of using the Pori agent framework."""

    # Set up the tool registry
    registry = ToolRegistry()
    register_all_tools(registry)

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # Create orchestrator
    orchestrator = Orchestrator(llm=llm, tools_registry=registry)

    # Define a simple task
    task = "Generate 3 random numbers between 1 and 100, then calculate the sum of the first N fibonacci numbers where N is the largest of those random numbers."

    print(f"🚀 Task: {task}")
    print("-" * 50)

    try:
        # Execute the task
        result = await orchestrator.execute_task(
            task=task,
            agent_settings=AgentSettings(max_steps=15),
        )

        print("\n" + "=" * 50)
        print("📊 RESULTS")
        print("=" * 50)

        if result["success"]:
            print("✅ Task completed successfully!")

            # Get the agent and final answer
            agent = result.get("agent")
            if agent:
                final_answer = agent.memory.get_final_answer()
                if final_answer:
                    print(f"\n📝 Final Answer: {final_answer['final_answer']}")
                    if final_answer.get("reasoning"):
                        print(f"💭 Reasoning: {final_answer['reasoning']}")

                print(f"\n📈 Steps taken: {result['steps_taken']}")
                print(f"🔧 Tool calls made: {len(agent.memory.tool_call_history)}")

                # Show tool call summary
                print("\n🛠️  Tool Call Summary:")
                for i, tool_call in enumerate(agent.memory.tool_call_history, 1):
                    status = "✅" if tool_call.success else "❌"
                    print(
                        f"  {i}. {status} {tool_call.tool_name}({tool_call.parameters})"
                    )
        else:
            print("❌ Task failed or incomplete")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
