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
from .memory import AgentMemory, create_memory_store
from .orchestrator import Orchestrator
from .team import Team, TeamMode
from .tools.registry import tool_registry
from .tools.standard import register_all_tools

# Configure logging
from .utils.file_refs import expand_file_refs
from .utils.logging_config import setup_logging
from .utils.prompt_loader import set_prompts_dir

loggers = setup_logging(level=logging.INFO, include_http=True)
logger = logging.getLogger("pori.main")


# Define some example tools


def _handle_cli_command(command: str, memory: AgentMemory) -> None:
    """Handle slash commands like /memory, /memory clear, etc."""
    parts = command.strip().split()
    cmd = parts[0].lower()

    if cmd in {"/new", "/reset", "/clear"}:
        # Start a fresh conversation: drop transient per-task state (messages,
        # tool-call history, open task records) so the next task doesn't
        # inherit prior-task context. Keep durable memory intact: experiences,
        # archival passages, core memory blocks, summaries.
        msg_count = len(memory.messages)
        tool_count = len(memory.tool_call_history)
        memory.messages.clear()
        memory.tool_call_history.clear()
        memory.tasks.clear()
        memory.current_task_id = None
        memory.state.clear()
        memory._persist()
        print(
            f"Cleared conversation ({msg_count} messages, "
            f"{tool_count} tool calls). Durable memory kept."
        )
        return

    if cmd == "/memory":
        subcmd = parts[1].lower() if len(parts) > 1 else "list"

        if subcmd == "list":
            print("\n=== Memory Overview ===")

            # Messages
            print(f"\nMessages: {len(memory.messages)}")
            for msg in memory.messages[-10:]:
                preview = msg.content[:80].replace("\n", " ")
                print(f"  [{msg.role}] {preview}...")

            # Experiences
            print(f"\nExperiences: {len(memory.experiences)}")
            for exp in memory.experiences:
                text = str(exp.get("text", ""))[:80].replace("\n", " ")
                imp = exp.get("importance", 1)
                print(f"  [imp={imp}] {text}")

            # Tasks
            print(f"\nTasks: {len(memory.tasks)}")
            for tid, task in memory.tasks.items():
                print(f"  [{task.status}] {tid}: {task.description[:60]}")

            # Tool calls
            print(f"\nTool calls: {len(memory.tool_call_history)}")

            # Core memory blocks
            if getattr(memory, "core_memory", None):
                print("\nCore memory blocks:")
                for label, block in memory.core_memory._blocks.items():
                    lines = block.value.strip()
                    if lines:
                        print(f"  [{label}] ({len(lines)} chars) {lines[:80]}...")
                    else:
                        print(f"  [{label}] (empty)")

            # Archival passages
            print(f"\nArchival passages: {len(memory.archival_passages)}")
            for rec in memory.archival_passages:
                text = str(rec.get("text", ""))[:80].replace("\n", " ")
                tags = rec.get("tags", [])
                print(f"  {text}" + (f"  tags={tags}" if tags else ""))

        elif subcmd == "clear":
            target = parts[2].lower() if len(parts) > 2 else "all"
            if target == "all":
                memory.messages.clear()
                memory.experiences.clear()
                memory.tasks.clear()
                memory.tool_call_history.clear()
                memory.summaries.clear()
                memory.archival_passages.clear()
                memory.state.clear()
                if getattr(memory, "core_memory", None):
                    for block in memory.core_memory._blocks.values():
                        block.value = ""
                memory._persist()
                print("All memory cleared.")
            elif target == "messages":
                memory.messages.clear()
                memory._persist()
                print("Messages cleared.")
            elif target == "experiences":
                memory.experiences.clear()
                memory._persist()
                print("Experiences cleared.")
            elif target == "tasks":
                memory.tasks.clear()
                memory._persist()
                print("Tasks cleared.")
            elif target == "archival":
                memory.archival_passages.clear()
                memory._persist()
                print("Archival passages cleared.")
            else:
                print(f"Unknown clear target: {target}")
                print("Usage: /memory clear [all|messages|experiences|tasks|archival]")

        else:
            print(f"Unknown subcommand: {subcmd}")
            print(
                "Usage: /memory [list|clear [all|messages|experiences|tasks|archival]]"
            )

    else:
        print(f"Unknown command: {cmd}")
        print("Available commands: /memory, /model, /new")


# Small curated per-provider lists used by /model. Users can also type a
# raw model id, so these don't need to be exhaustive.
_PROVIDER_MODELS: dict[str, list[str]] = {
    "anthropic": [
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5",
        "claude-opus-4-5",
        "claude-3-5-sonnet-20241022",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5-mini",
    ],
    "google": [
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
}

_PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _switch_model_interactive(current: "LLMConfig"):
    """Two-step picker: choose provider, then model. Rebuilds LLM via create_llm().

    Returns (llm, new_llm_config) on success, or (None, None) if the user
    cancels or an error occurs.
    """
    from .config import LLMConfig, create_llm
    from .llm import pick_openrouter_model

    providers = list(_PROVIDER_ENV_KEYS.keys())

    print("\n=== Select a provider ===", flush=True)
    for i, name in enumerate(providers, start=1):
        has_key = bool(os.getenv(_PROVIDER_ENV_KEYS[name]))
        tag = "" if has_key else "  (API key not set)"
        marker = " *" if name == current.provider else "  "
        print(f"  {i}.{marker}{name}{tag}")
    print(f"\n  Current: {current.provider} / {current.model}")
    print("  Press Enter to cancel.\n", flush=True)

    try:
        raw = input("  > ").strip().lower()
    except EOFError:
        return None, None
    if not raw:
        return None, None

    if raw.isdigit() and 1 <= int(raw) <= len(providers):
        provider = providers[int(raw) - 1]
    elif raw in providers:
        provider = raw
    else:
        print(f"  Unknown provider: {raw}")
        return None, None

    if not os.getenv(_PROVIDER_ENV_KEYS[provider]):
        print(
            f"  {_PROVIDER_ENV_KEYS[provider]} is not set; "
            f"export it before switching to {provider}."
        )
        return None, None

    # Choose model for the selected provider
    if provider == "openrouter":
        default = current.model if current.provider == "openrouter" else None
        model = pick_openrouter_model(default_slug=default)
    else:
        options = _PROVIDER_MODELS.get(provider, [])
        print(f"\nSelect a {provider} model:")
        for i, m in enumerate(options, start=1):
            print(f"  {i}. {m}")
        print("\n  Type a number, paste any model id, or press Enter to cancel.")
        try:
            raw_m = input("  > ").strip()
        except EOFError:
            return None, None
        if not raw_m:
            return None, None
        if raw_m.isdigit() and 1 <= int(raw_m) <= len(options):
            model = options[int(raw_m) - 1]
        else:
            model = raw_m

    new_cfg = LLMConfig(
        provider=provider,
        model=model,
        temperature=current.temperature,
        max_tokens=current.max_tokens,
        top_p=current.top_p,
        extra_params=dict(current.extra_params),
    )
    try:
        new_llm = create_llm(new_cfg)
    except Exception as e:
        print(f"  Failed to build LLM: {e}")
        return None, None
    return new_llm, new_cfg


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
        print(f"\nError: {e}")
        print("Please create a config.yaml file based on config.example.yaml")
        return
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        print(f"\nError initializing LLM: {e}")
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

    memory_backend = (
        getattr(config, "memory", None).backend
        if getattr(config, "memory", None)
        else "memory"
    )
    memory_sqlite_path = (
        getattr(config, "memory", None).sqlite_path
        if getattr(config, "memory", None)
        else None
    )
    memory_store = create_memory_store(
        backend=memory_backend,
        sqlite_path=memory_sqlite_path,
    )
    shared_memory = AgentMemory(
        user_id=getattr(config.memory, "user_id", "default_user"),
        agent_id=getattr(config.memory, "agent_id", "default_agent"),
        session_id=getattr(config.memory, "session_id", None),
        store=memory_store,
    )

    # Create orchestrator
    logger.info("Creating orchestrator")
    orchestrator = Orchestrator(
        llm=llm,
        tools_registry=registry,
        shared_memory=shared_memory,
    )

    # HITL: check if enabled in config
    hitl_handler = None
    hitl_config = getattr(config, "hitl", None)
    if hitl_config and hitl_config.enabled:
        hitl_handler = CLIHITLHandler(timeout_seconds=hitl_config.timeout_seconds)
        logger.info("HITL enabled in CLI mode")

    # Team: check if configured
    team_config = getattr(config, "team", None)
    use_team = team_config is not None and len(team_config.members) > 0
    if use_team:
        logger.info(
            f"Team mode enabled: '{team_config.name}' ({team_config.mode.value}) "
            f"with {len(team_config.members)} members"
        )
        print(
            f"(Team mode: {team_config.name} [{team_config.mode.value}] "
            f"with {len(team_config.members)} members)"
        )

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
    if memory_backend == "memory":
        print("(Memory backend: in-memory; exiting clears session memory.)")
    else:
        print(
            f"(Memory backend: {memory_backend}; session namespace={shared_memory.namespace})"
        )

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
            print("Goodbye!")
            break

        if not task:
            logger.warning("Empty task provided, skipping")
            continue

        # CLI commands
        if task.startswith("/"):
            if task.strip().split()[0].lower() == "/model":
                import sys as _sys

                _sys.stdout.flush()
                new_llm, new_llm_config = _switch_model_interactive(config.llm)
                _sys.stdout.flush()
                if new_llm is not None and new_llm_config is not None:
                    llm = new_llm
                    config.llm = new_llm_config
                    orchestrator.llm = llm
                    print(
                        "\n=== Model switched ===\n"
                        f"  Provider: {new_llm_config.provider}\n"
                        f"  Model:    {new_llm_config.model}\n"
                        "  Next task will use this model.\n"
                        "  Tip: type /new to start a fresh conversation "
                        "(drops prior-task context).\n",
                        flush=True,
                    )
                else:
                    print("  Model unchanged.", flush=True)
                continue

            _handle_cli_command(task, shared_memory)
            continue

        # Inline any @path file references the user typed (e.g. @main.py).
        expanded_task, ref_notes = expand_file_refs(task)
        if ref_notes:
            print("\nAttached files:")
            for note in ref_notes:
                print(note)
            logger.info(f"Expanded {len(ref_notes)} @path reference(s)")
        task = expanded_task

        logger.info("New task received")

        try:
            if use_team:
                # Execute via Team
                logger.info("Starting team execution")
                agent_defaults = AgentSettings(
                    max_steps=config.agent.max_steps,
                    context_window_tokens=config.agent.context_window_tokens,
                    context_window_reserve_tokens=config.agent.context_window_reserve_tokens,
                )
                team = Team(
                    task=task,
                    coordinator_llm=llm,
                    members=team_config.members,
                    mode=team_config.mode,
                    tools_registry=registry,
                    hitl_handler=hitl_handler,
                    hitl_config=hitl_config,
                    agent_defaults=agent_defaults,
                    max_delegation_steps=team_config.max_delegation_steps,
                    max_concurrent_members=team_config.max_concurrent_members,
                    name=team_config.name,
                )
                result = await team.run()

                logger.info(
                    f"Team execution completed - Success: {result['completed']}, Steps: {result['steps_taken']}"
                )

                print("\n=== Task Execution Summary ===")
                print(f"Task: {task}")
                print(f"Completed: {result['completed']}")
                print(f"Steps taken: {result['steps_taken']}")

                fs = result.get("final_state", {})
                if "plan_steps" in fs:
                    print(
                        f"Plan steps: {fs['plan_steps']} | Agent steps: {fs['agent_steps']}"
                    )
                if "chosen_member" in fs:
                    print(f"Routed to: {fs['chosen_member']}")

                final_answer = fs.get("final_answer")
                if final_answer:
                    print(f"\nFINAL ANSWER:\n  {final_answer}")
                else:
                    print("\nNO FINAL ANSWER FOUND")

            else:
                # Execute via single-agent Orchestrator
                logger.info("Starting task execution")
                result = await orchestrator.execute_task(
                    task=task,
                    agent_settings=AgentSettings(
                        max_steps=config.agent.max_steps,
                        context_window_tokens=config.agent.context_window_tokens,
                        context_window_reserve_tokens=config.agent.context_window_reserve_tokens,
                    ),
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
                        pass

                # Show the final answer if available
                agent = result.get("agent")

                if agent:
                    final_answer = agent.memory.get_final_answer()

                    if final_answer:
                        logger.info("Final answer provided")
                        print("\nFINAL ANSWER:")
                        print(f"  {final_answer['final_answer']}")
                        if final_answer.get("reasoning"):
                            print(f"\n  Reasoning: {final_answer['reasoning']}")
                        print(
                            "\n(Memory recall used if 'Retrieved knowledge' logs appear above for this task.)"
                        )
                    else:
                        logger.warning("No final answer found")
                        print("\nNO FINAL ANSWER FOUND")

                    calls_this_task = [
                        tc
                        for tc in agent.memory.tool_call_history
                        if getattr(tc, "task_id", None) == agent.task_id
                    ]
                    tool_calls_count = len(calls_this_task)
                    logger.info(f"Tool calls made (this task): {tool_calls_count}")

                    print("\nTool Calls (this task):")
                    for i, tool_call in enumerate(calls_this_task, start=1):
                        status = "+" if tool_call.success else "x"
                        print(
                            f"  {i}. {tool_call.tool_name}({tool_call.parameters}) -> {status}"
                        )

                        log_level = (
                            logging.INFO if tool_call.success else logging.WARNING
                        )
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
