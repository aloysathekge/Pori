"""Aloy MCP servers: scope/union resolver + CRUD endpoint gating."""

import pytest
from cryptography.fernet import Fernet

from aloy_backend import config as config_mod
from aloy_backend.models import ORG_CONNECTION_USER, McpServer

pytestmark = pytest.mark.asyncio

CLIENT_ORG = "user:test-user"
CLIENT_USER = "test-user"


@pytest.fixture(autouse=True)
def _enc_key(monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setattr(config_mod.settings, "connections_enc_key", key)
    from aloy_backend.connections import crypto

    crypto._fernet.cache_clear()
    yield
    crypto._fernet.cache_clear()


class TestResolver:
    async def test_union_and_user_precedence(self, db_session_maker):
        from aloy_backend.connections.mcp_store import resolve_run_mcp_servers

        async with db_session_maker() as session:
            # org-shared 'shared' + 'notes'; user's own 'notes' (same name)
            session.add(
                McpServer(
                    organization_id="biz",
                    user_id=ORG_CONNECTION_USER,
                    scope="org",
                    name="shared",
                    url="http://org-shared",
                )
            )
            session.add(
                McpServer(
                    organization_id="biz",
                    user_id=ORG_CONNECTION_USER,
                    scope="org",
                    name="notes",
                    url="http://org-notes",
                )
            )
            session.add(
                McpServer(
                    organization_id="biz",
                    user_id="alice",
                    scope="user",
                    name="notes",
                    url="http://alice-notes",
                )
            )
            await session.commit()
            configs = await resolve_run_mcp_servers(session, "biz", "alice")

        by_name = {c.name: c for c in configs}
        assert set(by_name) == {"shared", "notes"}
        # alice's own 'notes' wins over the org's
        assert by_name["notes"].url == "http://alice-notes"
        assert by_name["shared"].url == "http://org-shared"

    async def test_static_secret_becomes_bearer_header(self, db_session_maker):
        from aloy_backend.connections.crypto import encrypt
        from aloy_backend.connections.mcp_store import resolve_run_mcp_servers

        async with db_session_maker() as session:
            session.add(
                McpServer(
                    organization_id="biz",
                    user_id="alice",
                    scope="user",
                    name="secure",
                    url="http://x",
                    auth_kind="static",
                    static_secret_enc=encrypt("SECRET-KEY"),
                )
            )
            await session.commit()
            configs = await resolve_run_mcp_servers(session, "biz", "alice")
        assert configs[0].headers["Authorization"] == "Bearer SECRET-KEY"

    async def test_disabled_server_excluded(self, db_session_maker):
        from aloy_backend.connections.mcp_store import resolve_run_mcp_servers

        async with db_session_maker() as session:
            session.add(
                McpServer(
                    organization_id="biz",
                    user_id="alice",
                    scope="user",
                    name="off",
                    url="http://x",
                    enabled=False,
                )
            )
            await session.commit()
            configs = await resolve_run_mcp_servers(session, "biz", "alice")
        assert configs == []

    async def test_personal_server_private_across_members(self, db_session_maker):
        from aloy_backend.connections.mcp_store import resolve_run_mcp_servers

        async with db_session_maker() as session:
            session.add(
                McpServer(
                    organization_id="biz",
                    user_id="alice",
                    scope="user",
                    name="alice-only",
                    url="http://x",
                )
            )
            await session.commit()
            configs = await resolve_run_mcp_servers(session, "biz", "bob")
        assert configs == []


class TestEndpoints:
    async def test_create_and_list_user_server(self, client):
        resp = await client.post(
            "/v1/mcp-servers",
            json={"name": "notes", "url": "https://notes.example/mcp"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["scope"] == "user" and body["account_managed"] is True

        listing = await client.get("/v1/mcp-servers")
        assert any(s["name"] == "notes" for s in listing.json())

    async def test_duplicate_name_conflicts(self, client):
        payload = {"name": "dup", "url": "https://x.example/mcp"}
        assert (await client.post("/v1/mcp-servers", json=payload)).status_code == 201
        assert (await client.post("/v1/mcp-servers", json=payload)).status_code == 409

    async def test_static_auth_requires_secret(self, client):
        resp = await client.post(
            "/v1/mcp-servers",
            json={"name": "s", "url": "https://x/mcp", "auth_kind": "static"},
        )
        assert resp.status_code == 422

    async def test_owner_can_create_org_server(self, client):
        # the test client is owner of its personal org -> has CONNECTION_MANAGE
        resp = await client.post(
            "/v1/mcp-servers",
            json={"name": "orgsrv", "url": "https://x/mcp", "scope": "org"},
        )
        assert resp.status_code == 201
        assert resp.json()["scope"] == "org"

    async def test_toggle_and_delete(self, client):
        created = (
            await client.post(
                "/v1/mcp-servers", json={"name": "temp", "url": "https://x/mcp"}
            )
        ).json()
        sid = created["id"]
        patched = await client.patch(f"/v1/mcp-servers/{sid}", json={"enabled": False})
        assert patched.status_code == 200 and patched.json()["enabled"] is False
        assert (await client.delete(f"/v1/mcp-servers/{sid}")).status_code == 204
