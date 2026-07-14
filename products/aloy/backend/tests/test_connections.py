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


@pytest.fixture
def no_google_creds(monkeypatch):
    # Blank BOTH sources: process env and Settings (a dev box's real .env
    # feeds Settings, which the spec now falls back to).
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(config_mod.settings, "google_oauth_client_id", "")
    monkeypatch.setattr(config_mod.settings, "google_oauth_client_secret", "")


class TestProviders:
    def test_google_unconfigured_by_default(self, no_google_creds):
        from aloy_backend.connections import available_providers, get_provider

        assert get_provider("google") is not None
        assert available_providers() == []  # no creds -> not offered

    def test_google_available_when_configured(self, no_google_creds, monkeypatch):
        monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "id")
        monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "secret")
        from aloy_backend.connections import available_providers

        assert [p.name for p in available_providers()] == ["google"]

    def test_settings_fallback_when_env_unset(self, no_google_creds, monkeypatch):
        """pydantic-settings reads .env into Settings without touching
        os.environ — the spec must still see those credentials (the bug that
        hid the Connect button on any .env-configured deployment)."""
        monkeypatch.setattr(config_mod.settings, "google_oauth_client_id", "id")
        monkeypatch.setattr(config_mod.settings, "google_oauth_client_secret", "secret")
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


class TestScopeAndUnionResolver:
    async def test_org_scope_uses_sentinel_and_union_prefers_user(
        self, db_session_maker
    ):
        from aloy_backend.connections.crypto import encrypt
        from aloy_backend.connections.store import resolve_run_connections
        from aloy_backend.models import ORG_CONNECTION_USER

        async with db_session_maker() as session:
            # org-shared google (sentinel user), + the member's own google
            session.add(
                OAuthConnection(
                    organization_id="biz",
                    user_id=ORG_CONNECTION_USER,
                    scope="org",
                    provider="google",
                    access_token_enc=encrypt("ORG-TOKEN"),
                    account_email="shared@biz.com",
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                    status="active",
                )
            )
            session.add(
                OAuthConnection(
                    organization_id="biz",
                    user_id="alice",
                    scope="user",
                    provider="google",
                    access_token_enc=encrypt("ALICE-TOKEN"),
                    account_email="alice@gmail.com",
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                    status="active",
                )
            )
            await session.commit()
            ctx = await resolve_run_connections(session, "biz", "alice")
        # Alice's own token wins for google (her mailbox, not the org's).
        assert ctx["google"]["access_token"] == "ALICE-TOKEN"
        assert ctx["google"]["scope"] == "user"

    async def test_org_shared_used_when_member_has_none(self, db_session_maker):
        from aloy_backend.connections.crypto import encrypt
        from aloy_backend.connections.store import resolve_run_connections
        from aloy_backend.models import ORG_CONNECTION_USER

        async with db_session_maker() as session:
            session.add(
                OAuthConnection(
                    organization_id="biz",
                    user_id=ORG_CONNECTION_USER,
                    scope="org",
                    provider="google",
                    access_token_enc=encrypt("ORG-TOKEN"),
                    account_email="shared@biz.com",
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                    status="active",
                )
            )
            await session.commit()
            # bob has no personal google -> falls back to the org-shared one
            ctx = await resolve_run_connections(session, "biz", "bob")
        assert ctx["google"]["access_token"] == "ORG-TOKEN"
        assert ctx["google"]["scope"] == "org"

    async def test_personal_connection_private_across_members(self, db_session_maker):
        from aloy_backend.connections.crypto import encrypt
        from aloy_backend.connections.store import resolve_run_connections

        async with db_session_maker() as session:
            session.add(
                OAuthConnection(
                    organization_id="biz",
                    user_id="alice",
                    scope="user",
                    provider="google",
                    access_token_enc=encrypt("ALICE-TOKEN"),
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                    status="active",
                )
            )
            await session.commit()
            # bob must NOT see alice's personal connection
            ctx = await resolve_run_connections(session, "biz", "bob")
        assert ctx == {}


class TestEndpoints:
    async def test_list_empty_when_unconfigured(self, client, no_google_creds):
        resp = await client.get("/v1/connections")
        assert resp.status_code == 200
        assert resp.json() == []  # google not configured -> not offered

    async def test_start_404_when_provider_unconfigured(self, client, no_google_creds):
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


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _fake_http(monkeypatch, get_payloads=None, post_payload=None):
    """Patch google_common.httpx.Client; return the recorded calls list."""
    from aloy_backend.tools import google_common

    calls: list = []
    gets = list(get_payloads or [])

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def get(self, url, headers=None, params=None):
            calls.append(("GET", url, headers, params))
            return _FakeResp(gets.pop(0) if gets else {})

        def post(self, url, headers=None, json=None):
            calls.append(("POST", url, headers, json))
            return _FakeResp(post_payload or {})

    monkeypatch.setattr(google_common.httpx, "Client", FakeClient)
    return calls


CTX = {"connections": {"google": {"access_token": "TOK"}}}


class TestGoogleTools:
    def test_not_connected_message(self):
        from aloy_backend.tools.gmail import GmailSearchParams, gmail_search_tool

        out = gmail_search_tool(GmailSearchParams(query="x"), context={})
        assert "not connected" in out["error"].lower()

    def test_gmail_search_uses_injected_token(self, monkeypatch):
        from aloy_backend.tools.gmail import GmailSearchParams, gmail_search_tool

        calls = _fake_http(
            monkeypatch,
            get_payloads=[
                {"messages": [{"id": "m1"}]},
                {
                    "snippet": "hi",
                    "payload": {"headers": [{"name": "From", "value": "bob@x.com"}]},
                },
            ],
        )
        out = gmail_search_tool(GmailSearchParams(query="from:bob"), CTX)
        assert out["count"] == 1
        assert out["messages"][0]["from"] == "bob@x.com"
        assert any(h and h.get("Authorization") == "Bearer TOK" for _, _, h, _ in calls)

    def test_gmail_send(self, monkeypatch):
        from aloy_backend.tools.gmail import GmailSendParams, gmail_send_tool

        calls = _fake_http(monkeypatch, post_payload={"id": "sent1"})
        out = gmail_send_tool(
            GmailSendParams(to="a@b.com", subject="Hi", body="hello"), CTX
        )
        assert out == {"sent": True, "id": "sent1", "to": "a@b.com"}
        assert calls[0][0] == "POST" and calls[0][1].endswith("/messages/send")
        assert "raw" in calls[0][3]  # RFC822 base64 payload

    def test_gmail_create_draft(self, monkeypatch):
        from aloy_backend.tools.gmail import GmailDraftParams, gmail_create_draft_tool

        calls = _fake_http(
            monkeypatch, post_payload={"id": "d1", "message": {"id": "m1"}}
        )
        out = gmail_create_draft_tool(
            GmailDraftParams(to="a@b.com", subject="Hi", body="draft me"), CTX
        )
        assert out["drafted"] is True
        assert out["draft_id"] == "d1" and out["message_id"] == "m1"
        # Hits the drafts endpoint (NOT /messages/send) with a wrapped message.
        assert calls[0][0] == "POST" and calls[0][1].endswith("/drafts")
        assert "message" in calls[0][3] and "raw" in calls[0][3]["message"]

    def test_gmail_send_draft(self, monkeypatch):
        from aloy_backend.tools.gmail import GmailSendDraftParams, gmail_send_draft_tool

        calls = _fake_http(monkeypatch, post_payload={"id": "sent1"})
        out = gmail_send_draft_tool(GmailSendDraftParams(draft_id="d1"), CTX)
        assert out == {"sent": True, "id": "sent1", "draft_id": "d1"}
        # Sends the existing draft (drafts/send with its id) — no new compose.
        assert calls[0][0] == "POST" and calls[0][1].endswith("/drafts/send")
        assert calls[0][3] == {"id": "d1"}

    def test_gmail_list_drafts(self, monkeypatch):
        from aloy_backend.tools.gmail import (
            GmailListDraftsParams,
            gmail_list_drafts_tool,
        )

        calls = _fake_http(
            monkeypatch,
            get_payloads=[
                {"drafts": [{"id": "d1"}]},
                {
                    "message": {
                        "snippet": "hi",
                        "payload": {
                            "headers": [{"name": "Subject", "value": "Re: Hello"}]
                        },
                    }
                },
            ],
        )
        out = gmail_list_drafts_tool(GmailListDraftsParams(), CTX)
        assert out["count"] == 1
        assert out["drafts"][0]["draft_id"] == "d1"
        assert out["drafts"][0]["subject"] == "Re: Hello"
        assert calls[0][1].endswith("/drafts")

    def test_draft_gating_split(self):
        # Staging (create/list) has no external consequence → ungated. Delivery
        # (send/send_draft) is consequential → in the HITL-gate-able write set.
        from aloy_backend.tools.gmail import GMAIL_TOOL_NAMES, GMAIL_WRITE_TOOLS

        for staging in ("gmail_create_draft", "gmail_list_drafts"):
            assert staging in GMAIL_TOOL_NAMES
            assert staging not in GMAIL_WRITE_TOOLS
        assert GMAIL_WRITE_TOOLS == {"gmail_send", "gmail_send_draft"}

    def test_calendar_list(self, monkeypatch):
        from aloy_backend.tools.calendar import (
            CalendarListParams,
            calendar_list_events_tool,
        )

        _fake_http(
            monkeypatch,
            get_payloads=[
                {
                    "items": [
                        {
                            "id": "e1",
                            "summary": "Standup",
                            "start": {"dateTime": "2026-07-09T09:00:00Z"},
                            "end": {"dateTime": "2026-07-09T09:15:00Z"},
                        }
                    ]
                }
            ],
        )
        out = calendar_list_events_tool(CalendarListParams(max_results=5), CTX)
        assert out["count"] == 1
        assert out["events"][0]["summary"] == "Standup"

    def test_calendar_create(self, monkeypatch):
        from aloy_backend.tools.calendar import (
            CalendarCreateParams,
            calendar_create_event_tool,
        )

        calls = _fake_http(
            monkeypatch, post_payload={"id": "ev1", "htmlLink": "http://x"}
        )
        out = calendar_create_event_tool(
            CalendarCreateParams(
                summary="Lunch",
                start="2026-07-09T12:00:00Z",
                end="2026-07-09T13:00:00Z",
            ),
            CTX,
        )
        assert out["created"] is True and out["id"] == "ev1"
        assert calls[0][3]["summary"] == "Lunch"

    def test_register_defines_google_group(self):
        from aloy_backend.tools import GOOGLE_TOOL_NAMES, register_google_tools
        from pori.tools.registry import tool_registry

        reg = tool_registry()
        register_google_tools(reg)
        for name in GOOGLE_TOOL_NAMES:
            assert name in reg.tools
