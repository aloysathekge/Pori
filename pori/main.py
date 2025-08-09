import asyncio
import logging
import os
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modules
from langchain_anthropic import ChatAnthropic
from .tools import tool_registry
from .agent import Agent, AgentSettings
from .orchestrator import Orchestrator
from .tools_builtin import register_all_tools

# Configure logging
from .utils.logging_config import setup_logging

loggers = setup_logging(level=logging.INFO, include_http=True)
logger = logging.getLogger("pori.main")


# Define some example tools


async def main():
    logger.info("Starting Pori Agent System")

    # Set up the tool registry
    logger.info("Initializing tool registry")
    registry = tool_registry()

    register_all_tools(registry)
    logger.info(f"Registered {len(registry.tools)} tools")

    # Create LLM - uses ANTHROPIC_API_KEY from environment
    model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    logger.info(f"Initializing LLM with model: {model_name}")

    llm = ChatAnthropic(
        model=model_name,
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # Create orchestrator
    logger.info("Creating orchestrator")
    orchestrator = Orchestrator(llm=llm, tools_registry=registry)

    # Define steps callback for monitoring
    def on_step_end(agent: Agent):
        step_msg = f"Completed step {agent.state.n_steps}"
        print(f"Completed step {agent.state.n_steps}")
        logger.info(
            step_msg,
            extra={
                "task_id": getattr(agent, "task_id", "unknown"),
                "step": agent.state.n_steps,
            },
        )

    logger.info("Starting interactive loop")

    # Interactive loop for tasks
    while True:
        print("\n Pori at your service!")
        task = input(f"How can I help you today? enter q to exit \n").strip()

        # Exit if the user provides no task
        if task == "q":
            logger.info("User requested exit")
            print("Goodbye! 👋")
            break

        if not task:
            logger.warning("Empty task provided, skipping")
            continue

        logger.info(f"New task received: {task}")

        try:
            # Execute the task with the orchestrator
            logger.info("Starting task execution")
            result = await orchestrator.execute_task(
                task=task,
                agent_settings=AgentSettings(max_steps=10),
                on_step_end=on_step_end,
            )

            logger.info(
                f"Task execution completed - Success: {result['success']}, Steps: {result['steps_taken']}"
            )

            print("\n=== Task Execution Summary ===")
            print(f"Task: {task}")
            print(f"Success: {result['success']}")
            print(f"Steps taken: {result['steps_taken']}")

            # Show the final answer if available
            agent = result.get("agent")

            if agent:
                # Final answer is tracked per-task in memory.state; ensure we're in the right context
                final_answer = agent.memory.get_final_answer()

                if final_answer:
                    logger.info("Final answer provided")
                    print("\n📝 FINAL ANSWER:")
                    print(f"  {final_answer['final_answer']}")
                    if final_answer.get("reasoning"):
                        print(f"\n  Reasoning: {final_answer['reasoning']}")
                else:
                    logger.warning("No final answer found")
                    print("\n⚠️ NO FINAL ANSWER FOUND")

                # Show tool call history
                # Only show tool calls for this task
                calls_this_task = [
                    tc
                    for tc in agent.memory.tool_call_history
                    if getattr(tc, "task_id", None) == agent.task_id
                ]
                tool_calls_count = len(calls_this_task)
                logger.info(f"Tool calls made (this task): {tool_calls_count}")

                print("\nTool Calls (this task):")
                for i, tool_call in enumerate(calls_this_task, start=1):
                    status = "✓" if tool_call.success else "✗"
                    print(
                        f"  {i}. {tool_call.tool_name}({tool_call.parameters}) → {status}"
                    )

                    # Log each tool call
                    log_level = logging.INFO if tool_call.success else logging.WARNING
                    logger.log(
                        log_level,
                        f"Tool call: {tool_call.tool_name} - {'Success' if tool_call.success else 'Failed'}",
                    )

        except Exception as e:
            logger.error(f"Error executing task: {e}", exc_info=True)
            print(f"Error executing task: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise
