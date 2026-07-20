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
    FileBackend,
    LLMConfig,
    Orchestrator,
    RunProfile,
    SkillCatalog,
    create_llm,
    create_sandbox_provider,
    get_configured_llm,
    get_provider_profile,
    register_all_tools,
    set_prompts_dir,
    set_sandbox_provider,
    tool_registry,
)

from .config import settings
from .models import AgentConfig

logger = logging.getLogger("aloy_backend")

EVENT_SURFACE_ROUTING_PROMPT = """
You are working inside one durable Aloy Event. Decide from the user's meaning
and ongoing product need whether Conversation, a file, canonical Tasks, or an
interactive Event Surface is the right form of help. When the user wants a
recurring structured experience they can open and use over time—for example a
timetable, tracker, plan, map, dashboard, comparison workspace, or multi-view
operating screen—call request_event_surface with a concrete experience brief.
Do this even when the user does not know or say the term Surface. Do not create
Markdown, HTML, or a collection of Tasks as a substitute for an appropriate
Surface. Keep simple explanations and one-off outputs in Conversation or files.
A queued Surface request means building has started, not that a Surface is live;
never claim readiness until the host reports a verified publication.
""".strip()

EVENT_RESEARCH_ROUTING_PROMPT = """
When work depends on current public facts, perform sourced research rather than
answering from model memory. Use web_search to discover sources and
read_web_page for material claims; their evidence receipts are Event-scoped.
Persist reusable structured findings with event_record_upsert, citing those
evidence ids. Use event_records_list to inspect existing canonical findings
instead of guessing or recreating them. Mark unsupported or inaccessible claims
unverified instead of inventing them. If you create a Task whose definition of done requires current
web research and a cited report, set execution_profile=sourced_research; this is
a semantic decision expressed in the Task tool call, never a keyword rule.
""".strip()


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
    llm_config: Optional[LLMConfig] = None,
    allowed_tools: Optional[tuple[str, ...]] = None,
    denied_tools: tuple[str, ...] = (),
    allowed_capability_groups: Optional[tuple[str, ...]] = None,
    allowed_provider_profiles: Optional[tuple[str, ...]] = None,
    allowed_models: Optional[tuple[str, ...]] = None,
    skill_catalog: Optional[SkillCatalog] = None,
    run_profile: Optional[RunProfile] = None,
    file_backend: Optional[FileBackend] = None,
    enable_surface_requests: bool = False,
) -> Orchestrator:
    """
    Create an Orchestrator.

    Args:
        shared_memory: Optional AgentMemory seeded with conversation history.
        agent_config: Optional operator-managed legacy Conversation runtime.
        llm_config: Optional product-owned purpose model. Mutually exclusive
            with a legacy AgentConfig.
    """
    if not os.getenv("PORI_PROMPTS_DIR"):
        local_prompts = Path(__file__).resolve().parent / "prompts"
        if local_prompts.exists():
            set_prompts_dir(local_prompts)

    if agent_config is not None and llm_config is not None:
        raise ValueError("agent_config and llm_config are mutually exclusive")

    # Product purpose models are explicit and never inherit user prompt/tool
    # preferences. Ordinary Runs keep the existing AgentConfig/default path.
    if llm_config is not None:
        llm = create_llm(llm_config)
        provider = llm_config.provider
        model = llm_config.model
    elif agent_config:
        llm_config = LLMConfig(
            provider=agent_config.provider,
            model=agent_config.model,
            temperature=agent_config.temperature,
        )
        llm = create_llm(llm_config)
        provider = agent_config.provider
        model = agent_config.model
    else:
        llm, config = get_configured_llm()
        provider = config.llm.provider
        model = config.llm.model

    # Organization policy applies equally to a legacy AgentConfig and to
    # operator-selected purpose profiles that use the default model.
    if allowed_provider_profiles and provider not in allowed_provider_profiles:
        raise ValueError("Provider denied by current organization policy")
    if allowed_models and model not in allowed_models:
        raise ValueError("Model denied by current organization policy")

    model_capabilities = get_provider_profile(provider).capabilities

    # Build tool registry, optionally filtered
    registry = tool_registry()
    register_all_tools(registry)
    # Aloy product tools (Gmail, file library, …) added on top of the kernel
    # via its seam. Excluded per-run via denied_tools when unusable (no
    # connection / empty library).
    from .tools import (
        register_google_tools,
        register_library_tools,
        register_research_tools,
        register_surface_authoring_tools,
        register_surface_build_tools,
        register_surface_state_tools,
        register_task_tools,
    )
    from .tools.surface_requests import register_surface_request_tool

    register_google_tools(registry)
    register_library_tools(registry)
    register_research_tools(registry)
    register_task_tools(registry)
    register_surface_state_tools(registry)
    product_denied_tools = set(denied_tools)
    if enable_surface_requests:
        register_surface_request_tool(registry)
    else:
        product_denied_tools.add("request_event_surface")
        product_denied_tools.add("surface_state_read")
    if run_profile and run_profile.profile_id == "aloy.surface-builder":
        register_surface_authoring_tools(registry)
        register_surface_build_tools(registry)
    else:
        from .tools.surface_builds import SURFACE_BUILD_TOOL_NAMES
        from .tools.surfaces import SURFACE_AUTHORING_TOOL_NAMES

        product_denied_tools.update(SURFACE_AUTHORING_TOOL_NAMES)
        product_denied_tools.update(SURFACE_BUILD_TOOL_NAMES)

    # A purpose profile owns its executable tool contract. Legacy AgentConfig
    # tool preferences apply to ordinary runs, but cannot accidentally remove
    # tools required by a product-owned builder/bootstrap profile.
    configured_tools = (
        set(agent_config.tools or ())
        if agent_config is not None and run_profile is None
        else set()
    )
    if enable_surface_requests and configured_tools:
        # This is Event control-plane capability, not an optional user tool
        # preference. Organization policy may still deny it below.
        configured_tools.add("request_event_surface")
    requested_tools = configured_tools or None
    if allowed_tools:
        requested_tools = (
            requested_tools.intersection(allowed_tools)
            if requested_tools
            else set(allowed_tools)
        )
    registry = registry.filtered(
        include_tools=requested_tools,
        exclude_tools=product_denied_tools,
        include_groups=allowed_capability_groups,
    )
    if run_profile and run_profile.profile_id == "aloy.surface-builder":
        from .tools.surface_completion import (
            register_surface_builder_completion_tool,
        )

        register_surface_builder_completion_tool(registry)

    configured_system_prompt = agent_config.system_prompt if agent_config else None
    system_prompt_blocks = [
        block
        for block in (
            configured_system_prompt,
            EVENT_SURFACE_ROUTING_PROMPT if enable_surface_requests else None,
            EVENT_RESEARCH_ROUTING_PROMPT if enable_surface_requests else None,
        )
        if block
    ]

    return Orchestrator(
        llm=llm,
        tools_registry=registry,
        shared_memory=shared_memory,
        skill_catalog=skill_catalog,
        system_prompt="\n\n".join(system_prompt_blocks) or None,
        model_capabilities=model_capabilities,
        run_profile=run_profile,
        file_backend=file_backend,
    )
