"""Account connections: encryption, token store/refresh, endpoints, Gmail tools."""

from datetime import datetime, timedelta, timezone

import pytest
from cryptography.fernet import Fernet

from aloy_backend import config as config_mod
from aloy_backend.models import OAuthConnection

pytestmark = pytest.mark.asyncio

CLIENT_ORG = "user:test-user"
CLIENT_USER = "test-user"


@pytest.fixture(autouse=True)
def _enc_key(monkeypatch):
    # A valid Fernet key so crypto works; clear the lru_cache so it's picked up.
    key = Fernet.generate_key().decode()
    monkeypatch.setattr(config_mod.settings, "connections_enc_key", key)
    from aloy_backend.connections import crypto

    crypto._fernet.cache_clear()
    yield
    crypto._fernet.cache_clear()


class TestCrypto:
    def test_roundtrip(self):
        from aloy_backend.connections.crypto import decrypt, encrypt

        assert decrypt(encrypt("sk-secret-token")) == "sk-secret-token"

    def test_ciphertext_is_not_plaintext(self):
        from aloy_backend.connections.crypto import encrypt

        assert "sk-secret-token" not in encrypt("sk-secret-token")


class TestProviders:
    def test_google_unconfigured_by_default(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_ID", raising=False)
        monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_SECRET", raising=False)
        from aloy_backend.connections import available_providers, get_provider

        assert get_provider("google") is not None
        assert available_providers() == []  # no creds -> not offered

    def test_google_available_when_configured(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "id")
        monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "secret")
        from aloy_backend.connections import available_providers

        assert [p.name for p in available_providers()] == ["google"]


class TestTokenStore:
    async def test_get_access_token_decrypts(self, db_session_maker):
        from aloy_backend.connections.crypto import encrypt
        from aloy_backend.connections.providers import GOOGLE
        from aloy_backend.connections.store import get_access_token

        async with db_session_maker() as session:
            session.add(
                OAuthConnection(
                    organization_id=CLIENT_ORG,
                    user_id=CLIENT_USER,
                    provider="google",
                    access_token_enc=encrypt("live-token"),
                    scopes=["gmail.readonly"],
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                    status="active",
                )
            )
            await session.commit()
            token = await get_access_token(session, CLIENT_ORG, CLIENT_USER, GOOGLE)
        assert token == "live-token"

    async def test_no_connection_returns_none(self, db_session_maker):
        from aloy_backend.connections.providers import GOOGLE
        from aloy_backend.connections.store import get_access_token

        async with db_session_maker() as session:
            token = await get_access_token(session, CLIENT_ORG, CLIENT_USER, GOOGLE)
        assert token is None

    async def test_resolve_run_connections(self, db_session_maker):
        from aloy_backend.connections.crypto import encrypt
        from aloy_backend.connections.store import resolve_run_connections

        async with db_session_maker() as session:
            session.add(
                OAuthConnection(
                    organization_id=CLIENT_ORG,
                    user_id=CLIENT_USER,
                    provider="google",
                    access_token_enc=encrypt("tok"),
                    account_email="alice@gmail.com",
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                    status="active",
                )
            )
            await session.commit()
            ctx = await resolve_run_connections(session, CLIENT_ORG, CLIENT_USER)
        assert ctx["google"]["access_token"] == "tok"
        assert ctx["google"]["account_email"] == "alice@gmail.com"


class TestEndpoints:
    async def test_list_empty_when_unconfigured(self, client):
        resp = await client.get("/v1/connections")
        assert resp.status_code == 200
        assert resp.json() == []  # google not configured -> not offered

    async def test_start_404_when_provider_unconfigured(self, client):
        resp = await client.post("/v1/connections/google/start")
        assert resp.status_code == 404

    async def test_start_returns_authorize_url_when_configured(
        self, client, monkeypatch
    ):
        monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "the-client-id")
        monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "the-secret")
        resp = await client.post("/v1/connections/google/start")
        assert resp.status_code == 200
        url = resp.json()["authorize_url"]
        assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
        assert "code_challenge=" in url
        assert "the-client-id" in url
        assert "gmail.readonly" in url

    async def test_disconnect_404_when_not_connected(self, client):
        resp = await client.delete("/v1/connections/google")
        assert resp.status_code == 404


class TestGmailTools:
    def test_not_connected_message(self):
        from aloy_backend.tools.gmail import GmailSearchParams, gmail_search_tool

        out = gmail_search_tool(GmailSearchParams(query="x"), context={})
        assert "not connected" in out["error"].lower()

    def test_search_uses_injected_token(self, monkeypatch):
        from aloy_backend.tools import gmail

        calls = []

        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                # first call = list, subsequent = message metadata
                if calls[-1][0].endswith("/messages"):
                    return {"messages": [{"id": "m1"}]}
                return {
                    "snippet": "hi there",
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "bob@x.com"},
                            {"name": "Subject", "value": "Hello"},
                            {"name": "Date", "value": "Mon"},
                        ]
                    },
                }

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, headers=None, params=None):
                calls.append((url, headers))
                return FakeResp()

        monkeypatch.setattr(gmail.httpx, "Client", FakeClient)
        ctx = {"connections": {"google": {"access_token": "TOK"}}}
        out = gmail.gmail_search_tool(gmail.GmailSearchParams(query="from:bob"), ctx)
        assert out["count"] == 1
        assert out["messages"][0]["from"] == "bob@x.com"
        # the token was sent as a Bearer
        assert any(h and h.get("Authorization") == "Bearer TOK" for _, h in calls)

    def test_register_defines_google_group(self):
        from pori.tools.registry import tool_registry

        from aloy_backend.tools import register_gmail_tools

        reg = tool_registry()
        register_gmail_tools(reg)
        assert "gmail_search" in reg.tools
        assert "gmail_read" in reg.tools
