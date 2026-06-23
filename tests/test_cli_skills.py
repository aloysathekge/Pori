from pori import (
    AgentMemory,
    SkillCatalog,
    SkillManifest,
    load_skill_catalog_from_directories,
)
from pori.main import _handle_cli_command, _resolve_skill_command


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
