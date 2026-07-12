"""Skill import: SKILL.md preview parsing (url|text), slug sanitizing,
warnings, and the preview→create roundtrip. Plus the MCP test-connection
endpoint's connected/failed shapes."""

import pytest

import aloy_backend.routes.mcp_servers as mcp_routes
import aloy_backend.routes.skills as skills_routes
from aloy_backend.routes.skills import _github_raw, _slugify

pytestmark = pytest.mark.asyncio

SKILL_MD = """---
name: Weekly Review
description: Run a structured weekly review over tasks and goals.
version: "2"
tags: productivity, review
author: someone
---
# Weekly review

1. Collect open tasks
2. Score progress
"""


class TestSkillPreview:
    async def test_text_with_frontmatter_parses(self, client):
        resp = await client.post("/v1/skills/preview", json={"text": SKILL_MD})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["slug"] == "weekly-review"
        assert body["name"] == "Weekly Review"
        assert body["version"] == "2"
        assert body["summary"].startswith("Run a structured")
        assert body["tags"] == ["productivity", "review"]
        assert body["author"] == "someone"
        assert "Collect open tasks" in body["instructions"]
        assert body["warnings"] == []

    async def test_bare_markdown_infers_and_warns(self, client):
        resp = await client.post(
            "/v1/skills/preview",
            json={"text": "Always answer in haiku.\nMore detail here."},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary"] == "Always answer in haiku."
        assert len(body["warnings"]) == 2  # no frontmatter + inferred summary

    async def test_exactly_one_of_url_or_text(self, client):
        assert (await client.post("/v1/skills/preview", json={})).status_code == 422
        assert (
            await client.post(
                "/v1/skills/preview", json={"text": "x", "url": "https://e.x/s.md"}
            )
        ).status_code == 422

    async def test_empty_instructions_rejected(self, client):
        resp = await client.post(
            "/v1/skills/preview", json={"text": "---\nname: X\n---\n   "}
        )
        assert resp.status_code == 422

    async def test_url_flow_with_mocked_fetch(self, client, monkeypatch):
        async def fake_fetch(url: str) -> str:
            assert url == "https://example.com/skills/code-review.md"
            return SKILL_MD

        monkeypatch.setattr(skills_routes, "_fetch_skill_text", fake_fetch)
        resp = await client.post(
            "/v1/skills/preview",
            json={"url": "https://example.com/skills/code-review.md"},
        )
        assert resp.status_code == 200
        assert resp.json()["slug"] == "weekly-review"

    async def test_preview_then_create_roundtrip(self, client):
        preview = (
            await client.post("/v1/skills/preview", json={"text": SKILL_MD})
        ).json()
        preview.pop("warnings")
        created = await client.post("/v1/skills", json=preview)
        assert created.status_code == 201, created.text
        assert created.json()["slug"] == "weekly-review"


class TestHelpers:
    def test_slugify_shapes(self):
        assert _slugify("My Cool Skill!") == "my-cool-skill"
        assert _slugify("  --Weird---name--  ") == "weird-name"
        assert len(_slugify("x" * 200)) <= 64
        assert len(_slugify("a")) >= 3  # padded to satisfy the create pattern

    def test_github_blob_rewrite(self):
        assert (
            _github_raw("https://github.com/org/repo/blob/main/skills/x.md")
            == "https://raw.githubusercontent.com/org/repo/main/skills/x.md"
        )
        assert _github_raw("https://example.com/x.md") == "https://example.com/x.md"


class _FakeSessionSet:
    def __init__(self, configs, connected=True, tools=("mcp__t__alpha",)):
        self._configs = configs
        self._connected = connected
        self._tools = tools

    def connect_and_register(self, registry):
        if not self._connected:
            return 0
        for name in self._tools:
            registry.tools[name] = object()
        return len(self._tools)

    @property
    def connected_server_names(self):
        return [c.name for c in self._configs] if self._connected else []

    def close(self):
        pass


class TestMcpTestEndpoint:
    async def _make_server(self, client) -> str:
        resp = await client.post(
            "/v1/mcp-servers",
            json={
                "name": "probe-me",
                "url": "https://mcp.example.com/sse",
                "transport": "sse",
                "auth_kind": "none",
                "scope": "user",
            },
        )
        assert resp.status_code == 201, resp.text
        return resp.json()["id"]

    async def test_connected_reports_tools(self, client, monkeypatch):
        server_id = await self._make_server(client)
        monkeypatch.setattr(mcp_routes, "McpSessionSet", _FakeSessionSet)
        resp = await client.post(f"/v1/mcp-servers/{server_id}/test")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["tool_count"] == 1
        assert body["tools"] == ["mcp__t__alpha"]

    async def test_failed_connect_reports_cleanly(self, client, monkeypatch):
        server_id = await self._make_server(client)
        monkeypatch.setattr(
            mcp_routes,
            "McpSessionSet",
            lambda configs: _FakeSessionSet(configs, connected=False),
        )
        resp = await client.post(f"/v1/mcp-servers/{server_id}/test")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        assert "check the URL" in body["detail"]


class TestCreatedSkillsActuallyLoad:
    async def test_imported_skill_reaches_the_run_catalog(
        self, client, db_session_maker
    ):
        """The regression that mattered: created skills used to be draft +
        grantless — status=='approved' AND a grant are both required by
        load_skill_catalog, so no web-created skill ever loaded into a run."""
        preview = (
            await client.post("/v1/skills/preview", json={"text": SKILL_MD})
        ).json()
        preview.pop("warnings")
        created = await client.post("/v1/skills", json=preview)
        assert created.status_code == 201

        from aloy_backend.skills import load_skill_catalog
        from tests.conftest import TEST_USER_ID

        async with db_session_maker() as s:
            catalog = await load_skill_catalog(
                s,
                organization_id=f"user:{TEST_USER_ID}",
                user_id=TEST_USER_ID,
                role="owner",
            )
        slugs = [m.slug for m in catalog.manifests()]
        assert "weekly-review" in slugs
