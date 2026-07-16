from aloy_backend import orchestrator as orchestrator_module
from aloy_backend.models import AgentConfig
from pori import RunProfile


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
