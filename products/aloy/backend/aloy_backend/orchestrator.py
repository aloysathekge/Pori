import os
from pathlib import Path
from typing import Optional

from pori import (
    AgentMemory,
    Orchestrator,
    SkillCatalog,
    register_all_tools,
    tool_registry,
)
from pori.config import LLMConfig, create_llm, get_configured_llm
from pori.utils.prompt_loader import set_prompts_dir

from .models import AgentConfig


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
    # Aloy product tools (Gmail, …) added on top of the kernel via its seam.
    # They're excluded per-run via denied_tools when the user isn't connected.
    from .tools import register_google_tools

    register_google_tools(registry)

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
