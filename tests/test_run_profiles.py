from __future__ import annotations

import pytest
from pydantic import ValidationError

from pori import (
    AgentSettings,
    Orchestrator,
    RunProfile,
    RunProfileResolutionError,
    SkillCatalog,
    SkillManifest,
)


def _catalog(*slugs: str) -> SkillCatalog:
    catalog = SkillCatalog()
    for slug in slugs:
        catalog.register(
            SkillManifest(
                slug=slug,
                name=slug.replace("-", " ").title(),
                version="1",
                summary=f"Instructions for {slug}",
                required_model_capabilities=frozenset({"tools"}),
            ),
            f"Follow the {slug} contract.",
        )
    return catalog


def test_run_profile_fingerprint_is_order_independent_and_versioned():
    first = RunProfile(
        profile_id="aloy.surface-builder",
        allowed_tools=frozenset({"write", "read"}),
        required_skill_ids=frozenset({"critic@1", "builder@1"}),
    )
    reordered = RunProfile(
        profile_id="aloy.surface-builder",
        allowed_tools=frozenset({"read", "write"}),
        required_skill_ids=frozenset({"builder@1", "critic@1"}),
    )
    changed = reordered.model_copy(update={"version": "2"})

    assert first.fingerprint == reordered.fingerprint
    assert first.fingerprint != changed.fingerprint
    assert first.descriptor()["profile_id"] == "aloy.surface-builder"


def test_run_profile_rejects_contradictory_tool_contract():
    with pytest.raises(ValidationError, match="required_tools cannot also be denied"):
        RunProfile(
            profile_id="invalid",
            required_tools=frozenset({"write"}),
            denied_tools=frozenset({"write"}),
        )


def test_orchestrator_fails_fast_when_profile_requirements_are_missing(
    mock_llm, tool_registry
):
    profile = RunProfile(
        profile_id="aloy.surface-builder",
        required_skill_ids=frozenset({"surface-builder@1"}),
        required_model_capabilities=frozenset({"tools", "vision"}),
    )

    with pytest.raises(RunProfileResolutionError, match="vision"):
        Orchestrator(
            llm=mock_llm,
            tools_registry=tool_registry,
            skill_catalog=_catalog("surface-builder"),
            model_capabilities=frozenset({"tools"}),
            run_profile=profile,
        )

    with pytest.raises(RunProfileResolutionError, match="surface-builder@1"):
        Orchestrator(
            llm=mock_llm,
            tools_registry=tool_registry,
            model_capabilities=frozenset({"tools", "vision"}),
            run_profile=profile,
        )


@pytest.mark.asyncio
async def test_orchestrator_applies_profile_to_agent_and_result(
    mock_llm, tool_registry
):
    profile = RunProfile(
        profile_id="aloy.surface-builder",
        system_prompt="Preserve the last-good Surface revision.",
        allowed_tools=frozenset({"test_tool", "done"}),
        required_tools=frozenset({"test_tool"}),
        required_skill_ids=frozenset({"surface-builder@1"}),
        required_model_capabilities=frozenset({"tools"}),
    )
    orchestrator = Orchestrator(
        llm=mock_llm,
        tools_registry=tool_registry,
        skill_catalog=_catalog("surface-builder", "event-critic"),
        skill_limit=1,
        system_prompt="Respect the user's Event scope.",
        model_capabilities=frozenset({"tools"}),
        run_profile=profile,
    )

    assert set(orchestrator.tools_registry.tools) == {"test_tool", "done"}
    result = await orchestrator.execute_task(
        "Revise my university Surface",
        agent_settings=AgentSettings(max_steps=1),
        selected_skill_ids=["event-critic@1"],
    )

    agent = result["agent"]
    assert "Respect the user's Event scope." in agent.system_message
    assert "Preserve the last-good Surface revision." in agent.system_message
    assert agent.model_capabilities == frozenset({"tools"})
    assert {skill.manifest.skill_id for skill in agent.selected_skills} == {
        "surface-builder@1",
        "event-critic@1",
    }
    assert result["run_profile"] == profile.descriptor()
