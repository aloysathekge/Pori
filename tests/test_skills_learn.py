"""User-triggered skill authoring — /learn (SK-1, layer 1)."""

import pytest

from pori.skill_provenance import is_agent_created, use_write_origin
from pori.skills_learn import build_learn_prompt
from pori.tools.standard.core_tools import WriteSkillParams, write_skill_tool

pytestmark = [pytest.mark.unit]

VALID = """---
name: {slug}
description: Search arXiv papers by keyword, author, or ID.
version: 0.1.0
author: Pori
---

# Title
1. Use web_search to find the source.
"""


def test_build_learn_prompt_mentions_write_skill_and_focus():
    prompt = build_learn_prompt("the git release workflow")
    assert "write_skill" in prompt
    assert "git release workflow" in prompt
    assert "60" in prompt  # the <=60-char description rule


def test_build_learn_prompt_defaults_to_conversation():
    assert "conversation" in build_learn_prompt("").lower()


def test_write_skill_installs_valid_skill(tmp_path):
    res = write_skill_tool(
        WriteSkillParams(
            slug="search-arxiv", content=VALID.format(slug="search-arxiv")
        ),
        {"skills_dir": str(tmp_path)},
    )
    assert res["success"] is True
    dest = tmp_path / "search-arxiv" / "SKILL.md"
    assert dest.exists() and "Search arXiv" in dest.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "slug,content",
    [
        ("Bad Slug!", VALID.format(slug="x")),  # invalid slug
        ("x-skill", "no frontmatter here"),  # no frontmatter
        ("y-skill", "---\nversion: 1\n---\nbody"),  # missing name/description
    ],
)
def test_write_skill_rejects_invalid(tmp_path, slug, content):
    res = write_skill_tool(
        WriteSkillParams(slug=slug, content=content), {"skills_dir": str(tmp_path)}
    )
    assert res["success"] is False


def test_write_skill_requires_overwrite_to_replace(tmp_path):
    ctx = {"skills_dir": str(tmp_path)}
    body = VALID.format(slug="search-arxiv")
    assert write_skill_tool(WriteSkillParams(slug="search-arxiv", content=body), ctx)[
        "success"
    ]
    assert not write_skill_tool(
        WriteSkillParams(slug="search-arxiv", content=body), ctx
    )["success"]
    assert write_skill_tool(
        WriteSkillParams(slug="search-arxiv", content=body, overwrite=True), ctx
    )["success"]


def test_write_skill_marks_agent_created_only_under_agent_origin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # provenance ledger lives at ./.pori/skill_usage.json
    ctx = {"skills_dir": str(tmp_path / "skills")}

    write_skill_tool(
        WriteSkillParams(slug="user-skill", content=VALID.format(slug="user-skill")),
        ctx,
    )
    assert is_agent_created("user-skill@0.1.0") is False  # a user /learn is not curated

    with use_write_origin("background_review"):
        write_skill_tool(
            WriteSkillParams(
                slug="agent-skill", content=VALID.format(slug="agent-skill")
            ),
            ctx,
        )
    assert is_agent_created("agent-skill@0.1.0") is True
