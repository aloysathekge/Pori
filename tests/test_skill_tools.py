from pori import SkillCatalog, SkillManifest
from pori.tools.registry import ToolExecutor, ToolRegistry
from pori.tools.standard import register_all_tools


def _registry_with_skill_tools() -> ToolRegistry:
    registry = ToolRegistry()
    register_all_tools(registry)
    return registry


def test_skills_list_tool_searches_catalog_metadata():
    registry = _registry_with_skill_tools()
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="teach",
            name="Teach",
            version="1",
            summary="Teach the user a concept interactively",
            tags=("learning",),
            commands=("teach",),
            provenance="url",
            trust_level="untrusted",
            readiness_warnings=("review warning: contains 'system prompt:'",),
        ),
        "Teach well.",
    )
    snapshot = registry.snapshot()

    result = ToolExecutor(registry).execute_tool(
        "skills_list",
        {"query": "teach arithmetic", "limit": 5},
        {"skill_catalog": catalog, "capability_snapshot": snapshot},
    )

    assert result["success"] is True
    payload = result["result"]
    assert payload["available"] is True
    assert payload["skills"][0]["skill_id"] == "teach@1"
    assert payload["skills"][0]["score"] > 0
    assert payload["skills"][0]["provenance"] == "url"
    assert payload["skills"][0]["trust_level"] == "untrusted"
    assert payload["skills"][0]["readiness"] == "review_needed"
    assert "system prompt" in payload["skills"][0]["readiness_reasons"][0]


def test_skill_view_tool_loads_linked_files(tmp_path):
    skill_root = tmp_path / "teach"
    (skill_root / "references").mkdir(parents=True)
    (skill_root / "references" / "division.md").write_text(
        "Use sharing examples.",
        encoding="utf-8",
    )
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="teach",
            name="Teach",
            version="1",
            summary="Teach the user",
        ),
        "Main instructions.",
        root_path=skill_root,
    )
    registry = _registry_with_skill_tools()

    result = ToolExecutor(registry).execute_tool(
        "skill_view",
        {"skill": "teach", "file_path": "references/division.md"},
        {"skill_catalog": catalog, "capability_snapshot": registry.snapshot()},
    )

    assert result["success"] is True
    payload = result["result"]
    assert payload["available"] is True
    assert payload["skill_id"] == "teach@1"
    assert payload["path"] == "references/division.md"
    assert payload["content"] == "Use sharing examples."


def test_skills_list_omits_internal_model_invocation_flag():
    # The routing guard must not leak to the model (it makes the model wrongly
    # conclude a loaded skill is off-limits); descriptive metadata still shows.
    registry = _registry_with_skill_tools()
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="teach",
            name="Teach",
            version="1",
            summary="Teach the user a concept",
            model_invocation_disabled=True,
        ),
        "Teach well.",
    )

    result = ToolExecutor(registry).execute_tool(
        "skills_list",
        {"query": "teach"},
        {"skill_catalog": catalog, "capability_snapshot": registry.snapshot()},
    )

    skill = result["result"]["skills"][0]
    assert "model_invocation_disabled" not in skill
    assert skill["name"] == "Teach"
    assert skill["summary"]


def test_skill_view_signals_skill_is_loaded_and_usable():
    # A disable-model-invocation skill, once loaded, is still usable — the result
    # says so, so the model doesn't narrate "I can't use this".
    registry = _registry_with_skill_tools()
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="teach",
            name="Teach",
            version="1",
            summary="Teach the user",
            model_invocation_disabled=True,
        ),
        "Main instructions.",
    )

    result = ToolExecutor(registry).execute_tool(
        "skill_view",
        {"skill": "teach"},
        {"skill_catalog": catalog, "capability_snapshot": registry.snapshot()},
    )

    payload = result["result"]
    assert payload["loaded"] is True
    assert "loaded" in payload["usage"].lower()
    assert payload["content"] == "Main instructions."


def test_skill_tools_report_missing_catalog():
    registry = _registry_with_skill_tools()

    result = ToolExecutor(registry).execute_tool(
        "skills_list",
        {},
        {"capability_snapshot": registry.snapshot()},
    )

    assert result["success"] is True
    assert result["result"]["available"] is False
    assert "No skill catalog" in result["result"]["error"]
