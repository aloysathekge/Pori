"""The single place the backend constructs the Pori kernel:
``configure_sandbox`` activates the configured sandbox backend at process
startup (API server and worker both run agent code), ``sandbox_base_dir``
resolves the always-enforced filesystem jail root, and ``build_orchestrator``
assembles LLM, tool registry (kernel + Aloy product tools, filtered by org
policy and per-run denials), skills, and prompts into an ``Orchestrator``.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from pori import (
    AgentMemory,
    LLMConfig,
    Orchestrator,
    SkillCatalog,
    create_llm,
    create_sandbox_provider,
    get_configured_llm,
    register_all_tools,
    set_prompts_dir,
    set_sandbox_provider,
    tool_registry,
)

from .config import settings
from .models import AgentConfig

logger = logging.getLogger("aloy_backend")


def sandbox_base_dir() -> str:
    """The resolved filesystem jail root. Always available — even with the
    shell sandbox disabled, file tools confine writes to per-conversation
    dirs under this root instead of the (tenant-shared) host process cwd."""
    base = Path(settings.sandbox_base_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def configure_sandbox() -> None:
    """Point the kernel's sandbox provider at the configured backend, once at
    process startup (API server and worker both run agent code)."""
    if not settings.sandbox_enabled:
        return
    try:
        set_sandbox_provider(create_sandbox_provider(settings.sandbox_backend))
        logger.info("Sandbox backend active: %s", settings.sandbox_backend)
    except Exception:
        logger.exception(
            "Could not enable sandbox backend %r; agent code will run locally",
            settings.sandbox_backend,
        )


def build_orchestrator(
    shared_memory: Optional[AgentMemory] = None,
    agent_config: Optional[AgentConfig] = None,
    allowed_tools: Optional[tuple[str, ...]] = None,
    denied_tools: tuple[str, ...] = (),
    allowed_capability_groups: Optional[tuple[str, ...]] = None,
    allowed_provider_profiles: Optional[tuple[str, ...]] = None,
    allowed_models: Optional[tuple[str, ...]] = None,
    skill_catalog: Optional[SkillCatalog] = None,
) -> Orchestrator:
    """
    Create an Orchestrator.

    Args:
        shared_memory: Optional AgentMemory seeded with conversation history.
        agent_config: Optional per-user agent config (provider, model, tools, etc).
    """
    if not os.getenv("PORI_PROMPTS_DIR"):
        local_prompts = Path(__file__).resolve().parent / "prompts"
        if local_prompts.exists():
            set_prompts_dir(local_prompts)

    # Use agent config if provided, otherwise fall back to default config
    if agent_config:
        if (
            allowed_provider_profiles
            and agent_config.provider not in allowed_provider_profiles
        ):
            raise ValueError("Provider denied by current organization policy")
        if allowed_models and agent_config.model not in allowed_models:
            raise ValueError("Model denied by current organization policy")
        llm_config = LLMConfig(
            provider=agent_config.provider,
            model=agent_config.model,
            temperature=agent_config.temperature,
        )
        llm = create_llm(llm_config)
    else:
        llm, _cfg = get_configured_llm()

    # Build tool registry, optionally filtered
    registry = tool_registry()
    register_all_tools(registry)
    # Aloy product tools (Gmail, file library, …) added on top of the kernel
    # via its seam. Excluded per-run via denied_tools when unusable (no
    # connection / empty library).
    from .tools import (
        register_google_tools,
        register_library_tools,
        register_task_tools,
    )

    register_google_tools(registry)
    register_library_tools(registry)
    register_task_tools(registry)

    configured_tools = set(agent_config.tools or ()) if agent_config else set()
    requested_tools = configured_tools or None
    if allowed_tools:
        requested_tools = (
            requested_tools.intersection(allowed_tools)
            if requested_tools
            else set(allowed_tools)
        )
    registry = registry.filtered(
        include_tools=requested_tools,
        exclude_tools=denied_tools,
        include_groups=allowed_capability_groups,
    )

    return Orchestrator(
        llm=llm,
        tools_registry=registry,
        shared_memory=shared_memory,
        skill_catalog=skill_catalog,
    )
