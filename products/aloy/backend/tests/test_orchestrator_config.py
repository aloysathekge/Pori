from aloy_backend import orchestrator as orchestrator_module
from aloy_backend.models import AgentConfig
from aloy_backend.run_profiles import SURFACE_BUILDER_RUN_PROFILE
from aloy_backend.skills import _load_bundled_skill_catalog
from aloy_backend.tools.surface_builds import SURFACE_BUILD_TOOL_NAMES
from aloy_backend.tools.surfaces import SURFACE_AUTHORING_TOOL_NAMES
from pori import MemoryFileBackend, RunProfile


def test_build_orchestrator_threads_product_and_user_run_contract(monkeypatch):
    llm = object()
    monkeypatch.setattr(orchestrator_module, "create_llm", lambda _config: llm)
    agent_config = AgentConfig(
        organization_id="org-1",
        user_id="alice",
        name="Personal",
        provider="openai",
        model="gpt-4o",
        system_prompt="Use my preferred concise style.",
    )
    profile = RunProfile(
        profile_id="aloy.test-run",
        system_prompt="Follow the product run contract.",
        required_model_capabilities=frozenset({"tools", "vision"}),
    )

    orchestrator = orchestrator_module.build_orchestrator(
        agent_config=agent_config,
        run_profile=profile,
    )

    assert orchestrator.llm is llm
    assert orchestrator.model_capabilities == frozenset(
        {"tools", "structured_output", "vision"}
    )
    assert orchestrator.system_prompt == (
        "Use my preferred concise style.\n\nFollow the product run contract."
    )
    assert orchestrator.run_profile == profile


def test_surface_builder_orchestrator_is_explicit_and_file_scoped(monkeypatch):
    llm = object()
    monkeypatch.setattr(orchestrator_module, "create_llm", lambda _config: llm)
    agent_config = AgentConfig(
        organization_id="org-1",
        user_id="alice",
        name="Surface Builder",
        provider="openai",
        model="gpt-4o",
    )
    file_backend = MemoryFileBackend()

    orchestrator = orchestrator_module.build_orchestrator(
        agent_config=agent_config,
        run_profile=SURFACE_BUILDER_RUN_PROFILE,
        skill_catalog=_load_bundled_skill_catalog(),
        file_backend=file_backend,
    )

    assert SURFACE_AUTHORING_TOOL_NAMES | SURFACE_BUILD_TOOL_NAMES <= set(
        orchestrator.tools_registry.tools
    )
    assert "gmail_send" not in orchestrator.tools_registry.tools
    assert orchestrator.file_backend is file_backend
