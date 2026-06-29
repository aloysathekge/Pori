from pori import (
    AgentMemory,
    SkillCatalog,
    SkillLinkedFile,
    SkillManifest,
    load_skill_bundles_from_directory,
    load_skill_catalog_from_directories,
)
from pori.config import Config, LLMConfig, SkillsConfig
from pori.main import (
    _console_safe_text,
    _format_tool_call_parameters,
    _handle_cli_command,
    _handle_skills_lifecycle_command,
    _load_cli_skill_catalog,
    _missing_skill_argument_message,
    _resolve_auto_skill_selection,
    _resolve_skill_bundle_command,
    _resolve_skill_command,
    _resume_pending_skill_task,
    _summarize_loaded_skills,
    _summarize_written_artifacts,
)
from pori.memory import ToolCallRecord


def test_load_skill_catalog_from_local_skill_file(tmp_path, tool_registry):
    skill_dir = tmp_path / "skills" / "repo-workflow"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: Repo Workflow
slug: repo-workflow
version: 2
summary: Work safely in a repository
tags: [repo, workflow]
category: software-development
author: Pori
license: MIT
commands: [inspect, verify]
source_url: https://example.com/repo-workflow
install_command: pori skills install repo-workflow
argument-hint: Which repository task should this skill perform?
provenance: url
trust_level: untrusted
required_tools: [test_tool]
prerequisites:
  env_vars: [PORI_TEST_MISSING_SECRET]
  commands: [pori-missing-command]
setup:
  help: Install pori-missing-command and configure PORI_TEST_MISSING_SECRET.
---

# Repo Workflow

Always inspect source-of-truth files before editing.
""",
        encoding="utf-8",
    )

    catalog = load_skill_catalog_from_directories([tmp_path / "skills"])

    manifests = catalog.manifests()
    assert len(manifests) == 1
    assert manifests[0].skill_id == "repo-workflow@2"
    assert manifests[0].source.startswith("local:")
    assert manifests[0].required_tools == frozenset({"test_tool"})
    assert manifests[0].category == "software-development"
    assert manifests[0].author == "Pori"
    assert manifests[0].license == "MIT"
    assert manifests[0].commands == ("inspect", "verify")
    assert manifests[0].source_url == "https://example.com/repo-workflow"
    assert manifests[0].install_command == "pori skills install repo-workflow"
    assert manifests[0].provenance == "url"
    assert manifests[0].trust_level == "untrusted"
    assert manifests[0].required_credentials == ("PORI_TEST_MISSING_SECRET",)
    assert manifests[0].required_commands == ("pori-missing-command",)
    assert (
        manifests[0].setup_help
        == "Install pori-missing-command and configure PORI_TEST_MISSING_SECRET."
    )
    assert (
        manifests[0].argument_hint == "Which repository task should this skill perform?"
    )
    readiness = catalog.readiness("repo-workflow@2")
    assert readiness.status == "setup_needed"
    assert readiness.missing_credentials == ("PORI_TEST_MISSING_SECRET",)
    assert readiness.missing_commands == ("pori-missing-command",)

    summaries = catalog.summaries(tool_registry.snapshot())
    assert summaries[0].eligible is False
    assert summaries[0].reasons == ("missing_credential:PORI_TEST_MISSING_SECRET",)

    selected = catalog.load("repo-workflow@2")
    assert "Always inspect source-of-truth" in selected.instructions
    assert "required_tools" not in selected.instructions


def test_cli_summarizes_written_artifacts_and_redacts_content():
    calls = [
        ToolCallRecord(
            tool_name="sandbox_write_file",
            parameters={
                "path": "/mnt/user-data/workspace/lessons/division.html",
                "content": "<html>lesson</html>",
            },
            result={
                "success": True,
                "path": "/mnt/user-data/workspace/lessons/division.html",
                "bytes_written": 19,
            },
            success=True,
            task_id="task_1",
        )
    ]

    artifacts = _summarize_written_artifacts(calls)
    params = _format_tool_call_parameters(calls[0].parameters)

    assert artifacts == [
        "sandbox_write_file: wrote /mnt/user-data/workspace/lessons/division.html (19 bytes)"
    ]
    assert "'content': '<19 chars>'" in params
    assert "<html>lesson</html>" not in params


def test_cli_summarizes_runtime_loaded_skills():
    calls = [
        ToolCallRecord(
            tool_name="skill_view",
            parameters={"skill": "teach@1"},
            result={"success": True, "result": {"skill_id": "teach@1"}},
            success=True,
            task_id="task_1",
        ),
        ToolCallRecord(
            tool_name="skill_view",
            parameters={"skill": "teach@1"},
            result={"success": True, "result": {"skill_id": "teach@1"}},
            success=True,
            task_id="task_1",
        ),
    ]

    assert _summarize_loaded_skills(calls) == ["teach@1"]


def test_console_safe_text_replaces_unprintable_characters_for_legacy_codepages():
    assert _console_safe_text("12 − 3 = 9 ÷ 3", encoding="cp1252") == "12 ? 3 = 9 ÷ 3"


def test_skill_catalog_builds_hermes_style_index_and_searches(tool_registry):
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="teach",
            name="Teach",
            version="1",
            summary="Teach the user a new skill or concept, within this workspace.",
            tags=("learning", "productivity"),
            category="productivity",
            commands=("teach", "lesson"),
            source="local:.pori/skills/teach",
            install_command="copy teach into .pori/skills",
        ),
        "Teach interactively.",
    )

    index = catalog.index(tool_registry.snapshot())
    hits = catalog.search(
        "teach me multiplication like I am starting from zero",
        tool_registry.snapshot(),
    )

    assert index[0].skill_id == "teach@1"
    assert index[0].category == "productivity"
    assert index[0].install_command == "copy teach into .pori/skills"
    assert hits[0].entry.skill_id == "teach@1"
    assert "teach" in hits[0].matched_terms


def test_skill_loader_skips_disabled_skills_and_injects_declared_config(tmp_path):
    enabled_dir = tmp_path / "skills" / "repo-workflow"
    enabled_dir.mkdir(parents=True)
    (enabled_dir / "SKILL.md").write_text(
        """---
name: Repo Workflow
description: Work safely in a repository
metadata:
  pori:
    config:
      - key: github.default_owner
        description: Default GitHub owner/org
        default: fallback-owner
---

Use the configured GitHub owner when relevant.
""",
        encoding="utf-8",
    )
    disabled_dir = tmp_path / "skills" / "old-skill"
    disabled_dir.mkdir(parents=True)
    (disabled_dir / "SKILL.md").write_text(
        """---
name: Old Skill
description: Disabled skill
---

Do not load me.
""",
        encoding="utf-8",
    )

    catalog = load_skill_catalog_from_directories(
        [tmp_path / "skills"],
        disabled=["old-skill"],
        config_values={"github": {"default_owner": "aloysathekge"}},
    )

    assert [manifest.slug for manifest in catalog.manifests()] == ["repo-workflow"]
    selected = catalog.load("repo-workflow@1")
    assert "github.default_owner = aloysathekge" in selected.instructions


def test_skill_catalog_indexes_and_views_linked_files(tmp_path):
    skill_dir = tmp_path / "skills" / "brainstorming"
    (skill_dir / "references" / "old-package").mkdir(parents=True)
    (skill_dir / "scripts").mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: Brainstorming
description: Explore approaches before implementation
---

Ask clarifying questions before coding.
""",
        encoding="utf-8",
    )
    (skill_dir / "visual-companion.md").write_text(
        "Use diagrams when helpful.",
        encoding="utf-8",
    )
    (skill_dir / "scripts" / "review.py").write_text(
        "print('review')",
        encoding="utf-8",
    )
    (skill_dir / "references" / "old-package" / "SKILL.md").write_text(
        "This is documentation, not an active skill.",
        encoding="utf-8",
    )

    catalog = load_skill_catalog_from_directories([tmp_path / "skills"])

    assert [manifest.slug for manifest in catalog.manifests()] == ["brainstorming"]
    assert catalog.linked_files("brainstorming@1") == (
        SkillLinkedFile(path="scripts/review.py", kind="scripts", size_bytes=15),
        SkillLinkedFile(path="visual-companion.md", kind="file", size_bytes=26),
    )
    view = catalog.view_file("brainstorming@1", "visual-companion.md")
    assert view.content == "Use diagrams when helpful."
    assert view.path == "visual-companion.md"


def test_skill_catalog_rejects_linked_file_path_traversal(tmp_path):
    skill_dir = tmp_path / "skills" / "brainstorming"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: Brainstorming
description: Explore approaches
---

Ask first.
""",
        encoding="utf-8",
    )
    catalog = load_skill_catalog_from_directories([tmp_path / "skills"])

    try:
        catalog.view_file("brainstorming@1", "../secret.txt")
    except ValueError as exc:
        assert "within the skill directory" in str(exc)
    else:
        raise AssertionError("Expected traversal to be rejected")


def test_cli_skill_catalog_uses_default_project_skills_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skill_dir = tmp_path / ".pori" / "skills" / "local-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: Local Skill
description: Default local skill
---

Use local project knowledge.
""",
        encoding="utf-8",
    )
    config = Config(
        llm=LLMConfig(provider="anthropic", model="test-model"),
        skills=SkillsConfig(),
    )

    catalog = _load_cli_skill_catalog(config)

    assert catalog is not None
    assert [manifest.slug for manifest in catalog.manifests()] == ["local-skill"]


def test_skills_command_prints_catalog(capsys, tool_registry):
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="repo-workflow",
            name="Repo Workflow",
            version="1",
            summary="Work safely in a repository",
            tags=("repo",),
            required_tools=frozenset({"test_tool"}),
        ),
        "Follow repository workflow rules.",
    )

    _handle_cli_command(
        "/skills",
        AgentMemory(),
        skill_catalog=catalog,
        registry=tool_registry,
    )

    output = capsys.readouterr().out
    assert "repo-workflow@1" in output
    assert "available" in output
    assert "source:" in output


def test_skills_command_searches_catalog(capsys, tool_registry):
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="teach",
            name="Teach",
            version="1",
            summary="Teach the user a new skill or concept",
            tags=("learning",),
            install_command="copy teach into .pori/skills",
        ),
        "Teach interactively.",
    )

    _handle_cli_command(
        "/skills multiplication teach",
        AgentMemory(),
        skill_catalog=catalog,
        registry=tool_registry,
    )

    output = capsys.readouterr().out
    assert "=== Skill Search: multiplication teach ===" in output
    assert "teach@1" in output
    assert "install: copy teach into .pori/skills" in output


def test_skill_command_prints_detail_and_linked_file(capsys, tmp_path, tool_registry):
    skill_dir = tmp_path / "skills" / "brainstorming"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: Brainstorming
description: Explore approaches before implementation
---

Ask clarifying questions.
""",
        encoding="utf-8",
    )
    (skill_dir / "visual-companion.md").write_text(
        "Use a sketch.",
        encoding="utf-8",
    )
    catalog = load_skill_catalog_from_directories([tmp_path / "skills"])

    _handle_cli_command(
        "/skill brainstorming",
        AgentMemory(),
        skill_catalog=catalog,
        registry=tool_registry,
    )
    detail = capsys.readouterr().out
    assert "Linked files:" in detail
    assert "visual-companion.md" in detail

    _handle_cli_command(
        "/skill brainstorming visual-companion.md",
        AgentMemory(),
        skill_catalog=catalog,
        registry=tool_registry,
    )
    file_view = capsys.readouterr().out
    assert "Use a sketch." in file_view


def test_skills_lifecycle_installs_inspects_and_uninstalls_local_package(
    capsys, tmp_path, tool_registry
):
    source = tmp_path / "source" / "teach"
    (source / "references").mkdir(parents=True)
    (source / "SKILL.md").write_text(
        """---
name: Teach
description: Teach concepts interactively
version: 2
---

Guide the learner.
""",
        encoding="utf-8",
    )
    (source / "references" / "division.md").write_text(
        "Division examples.",
        encoding="utf-8",
    )
    install_dir = tmp_path / "project" / ".pori" / "skills"
    config = Config(
        llm=LLMConfig(provider="anthropic", model="test-model"),
        skills=SkillsConfig(default_dir=str(install_dir)),
    )

    inspected = _handle_skills_lifecycle_command(
        f"/skills inspect {source}",
        config=config,
        skill_catalog=None,
        registry=tool_registry,
    )
    inspect_output = capsys.readouterr().out
    assert inspected is True
    assert "Teach (teach@2)" in inspect_output
    assert "Support files: yes" in inspect_output

    installed = _handle_skills_lifecycle_command(
        f"/skills install {source}",
        config=config,
        skill_catalog=None,
        registry=tool_registry,
    )
    install_output = capsys.readouterr().out
    assert installed is True
    assert "Installed Teach (teach@2)" in install_output
    assert (install_dir / "teach" / "SKILL.md").exists()
    assert (install_dir / "teach" / "references" / "division.md").exists()

    catalog = load_skill_catalog_from_directories([install_dir])
    assert [manifest.skill_id for manifest in catalog.manifests()] == ["teach@2"]

    uninstalled = _handle_skills_lifecycle_command(
        "/skills uninstall teach",
        config=config,
        skill_catalog=catalog,
        registry=tool_registry,
    )
    uninstall_output = capsys.readouterr().out
    assert uninstalled is True
    assert "Uninstalled teach" in uninstall_output
    assert not (install_dir / "teach").exists()


def test_resolve_skill_command_matches_slug_and_preserves_task():
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="repo-workflow",
            name="Repo Workflow",
            version="1",
            summary="Work safely in a repository",
        ),
        "Follow repository workflow rules.",
    )

    assert _resolve_skill_command(catalog, "/repo-workflow inspect files") == (
        "repo-workflow@1",
        "inspect files",
    )
    assert _resolve_skill_command(catalog, "/missing inspect files") is None


def test_skill_bundle_catalog_loads_yaml_and_resolves_command(tmp_path):
    bundle_dir = tmp_path / "skill-bundles"
    bundle_dir.mkdir()
    (bundle_dir / "backend-dev.yaml").write_text(
        """name: Backend Dev
description: Backend workflow bundle
skills:
  - repo-workflow
  - test-driven
instruction: Keep changes focused.
""",
        encoding="utf-8",
    )
    bundles = load_skill_bundles_from_directory(bundle_dir)
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="repo-workflow",
            name="Repo Workflow",
            version="1",
            summary="Work safely in a repo",
        ),
        "Follow repo rules.",
    )
    catalog.register(
        SkillManifest(
            slug="test-driven",
            name="Test Driven",
            version="1",
            summary="Write tests first",
        ),
        "Use tests.",
    )

    resolved = _resolve_skill_bundle_command(
        catalog,
        bundles,
        "/backend-dev implement feature",
    )

    assert [bundle.slug for bundle in bundles.bundles()] == ["backend-dev"]
    assert resolved == (
        ["repo-workflow@1", "test-driven@1"],
        "implement feature",
    )


def test_auto_skill_selection_does_not_activate_natural_teaching_request(tool_registry):
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="teach",
            name="teach",
            version="1",
            summary="Teach the user a new skill or concept, within this workspace.",
        ),
        "Teach the user interactively.",
    )

    assert (
        _resolve_auto_skill_selection(
            catalog,
            tool_registry,
            "teach me multiplication like I am starting from zero",
            skill_limit=3,
        )
        == ()
    )


def test_missing_skill_argument_message_is_metadata_driven():
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="workshop",
            name="Workshop",
            version="1",
            summary="Run a structured workshop",
            commands=("workshop",),
            argument_hint="What topic should the workshop cover?",
        ),
        "Run the workshop.",
    )

    message = _missing_skill_argument_message(
        catalog,
        ["workshop@1"],
        "please workshop something",
    )

    assert message is not None
    assert "workshop@1: What topic should the workshop cover?" in message
    assert (
        _missing_skill_argument_message(
            catalog,
            ["workshop@1"],
            "please workshop API design",
        )
        is None
    )


def test_pending_skill_invocation_resumes_with_user_detail():
    catalog = SkillCatalog()
    catalog.register(
        SkillManifest(
            slug="teach",
            name="Teach",
            version="1",
            summary="Teach the user a new skill or concept",
            argument_hint="What would you like to learn about?",
        ),
        "Teach interactively.",
    )
    selected = ["teach@1"]
    pending_task = "I want you to teach me something"

    resumed = _resume_pending_skill_task(pending_task, "basic 1+1")

    assert resumed == "I want you to teach me basic 1+1"
    assert _missing_skill_argument_message(catalog, selected, resumed) is None
    assert catalog.build_invocation("teach@1", resumed).invocation_text == "basic 1+1"
