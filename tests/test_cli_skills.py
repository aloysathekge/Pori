from pori import (
    AgentMemory,
    SkillCatalog,
    SkillLinkedFile,
    SkillManifest,
    load_skill_catalog_from_directories,
)
from pori.config import Config, LLMConfig, SkillsConfig
from pori.main import (
    _handle_cli_command,
    _load_cli_skill_catalog,
    _resolve_skill_command,
)


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
required_tools: [test_tool]
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

    summaries = catalog.summaries(tool_registry.snapshot())
    assert summaries[0].eligible is True

    selected = catalog.load("repo-workflow@2")
    assert "Always inspect source-of-truth" in selected.instructions
    assert "required_tools" not in selected.instructions


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
