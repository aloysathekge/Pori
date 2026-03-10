import asyncio
import logging
import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Import modules
from pathlib import Path

from .agent import Agent, AgentSettings
from .config import get_configured_llm
from .hitl import CLIHITLHandler
from .orchestrator import Orchestrator
from .tools.registry import tool_registry
from .tools.standard import register_all_tools

# Configure logging
from .utils.logging_config import setup_logging
from .utils.prompt_loader import set_prompts_dir

loggers = setup_logging(level=logging.INFO, include_http=True)
logger = logging.getLogger("pori.main")


# Define some example tools


async def main():
    logger.info("Starting Pori Agent System")

    # Set up the tool registry
    logger.info("Initializing tool registry")
    registry = tool_registry()

    register_all_tools(registry)
    # Register sandbox tools (bash) so they appear in the same registry
    import pori.sandbox.sandbox_tools  # noqa: F401

    logger.info(f"Registered {len(registry.tools)} tools")

    # Create LLM from config file
    logger.info("Loading LLM configuration")
    try:
        llm, config = get_configured_llm()
        logger.info(
            f"Initialized LLM - Provider: {config.llm.provider}, Model: {config.llm.model}"
        )
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\n❌ Error: {e}")
        print("Please create a config.yaml file based on config.example.yaml")
        return
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        print(f"\n❌ Error initializing LLM: {e}")
        return

    # Prompts: allow overriding packaged prompt templates
    try:
        if (
            getattr(config, "prompts", None)
            and config.prompts
            and config.prompts.base_dir
        ):
            set_prompts_dir(config.prompts.base_dir)
            logger.info(f"Prompts override enabled; base_dir={config.prompts.base_dir}")
    except Exception:
        pass

    # Sandbox: if enabled, set provider and base dir for per-task workspace
    sandbox_base_dir = None
    if (
        getattr(config, "sandbox", None)
        and config.sandbox.enabled
        and config.sandbox.base_dir
    ):
        from pori.sandbox import LocalSandboxProvider, set_sandbox_provider

        set_sandbox_provider(LocalSandboxProvider())
        sandbox_base_dir = str(Path(config.sandbox.base_dir).resolve())
        logger.info(f"Sandbox enabled; base_dir={sandbox_base_dir}")

    # Create orchestrator
    logger.info("Creating orchestrator")
    orchestrator = Orchestrator(llm=llm, tools_registry=registry)

    # HITL: check if enabled in config
    hitl_handler = None
    hitl_config = getattr(config, "hitl", None)
    if hitl_config and hitl_config.enabled:
        hitl_handler = CLIHITLHandler(timeout_seconds=hitl_config.timeout_seconds)
        logger.info("HITL enabled in CLI mode")

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
    print("(Memory is kept for this session only; exit and restart clears it.)")

    # Interactive loop for tasks
    while True:
        print("\n Pori at your service!")
        try:
            task = input(f"How can I help you today? enter q to exit \n").strip()
        except EOFError:
            logger.info("Input closed (EOF)")
            print("\nGoodbye!")
            break

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
                agent_settings=AgentSettings(max_steps=config.agent.max_steps),
                on_step_end=on_step_end,
                sandbox_base_dir=sandbox_base_dir,
                hitl_handler=hitl_handler,
                hitl_config=hitl_config,
            )

            logger.info(
                f"Task execution completed - Success: {result['success']}, Steps: {result['steps_taken']}"
            )

            print("\n=== Task Execution Summary ===")
            print(f"Task: {task}")
            print(f"Success: {result['success']}")
            print(f"Steps taken: {result['steps_taken']}")

            # Show basic run metrics if available
            metrics = result.get("result", {}).get("metrics")
            if metrics:
                try:
                    tokens = metrics.get("tokens", {}) or {}
                    cost = metrics.get("cost_usd")
                    print(
                        f"Run metrics: duration={metrics.get('duration')}, "
                        f"steps={metrics.get('steps')}, "
                        f"llm_calls={metrics.get('llm_calls')}, "
                        f"tool_calls={metrics.get('tool_calls')}, "
                        f"tokens_in={tokens.get('input')}, "
                        f"tokens_out={tokens.get('output')}, "
                        f"tokens_total={tokens.get('total')}"
                    )
                    if cost is not None:
                        print(f"Estimated cost: {cost}")
                except Exception:
                    # Metrics are optional; ignore if shape is unexpected
                    pass

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
                    # Show memory recall snippet if it was used in this step (logged by agent)
                    print(
                        "\n(Memory recall used if 'Retrieved knowledge' logs appear above for this task.)"
                    )
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
