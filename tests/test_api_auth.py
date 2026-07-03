"""API key auth fails closed (no keys configured => deny, not allow)."""

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402

import pori.api.security as security  # noqa: E402

pytestmark = [pytest.mark.integration]


def test_no_keys_configured_fails_closed(monkeypatch):
    monkeypatch.setattr(security, "VALID_API_KEYS", [])
    monkeypatch.delenv("PORI_API_ALLOW_UNAUTHENTICATED", raising=False)
    with pytest.raises(HTTPException) as exc:
        security.get_api_key("anything")
    assert exc.value.status_code == 503  # not fail-open


def test_explicit_opt_in_allows_unauthenticated(monkeypatch):
    monkeypatch.setattr(security, "VALID_API_KEYS", [])
    monkeypatch.setenv("PORI_API_ALLOW_UNAUTHENTICATED", "1")
    assert security.get_api_key("x") == "x"


def test_valid_key_is_accepted(monkeypatch):
    monkeypatch.setattr(security, "VALID_API_KEYS", ["secret"])
    assert security.get_api_key("secret") == "secret"


@pytest.mark.parametrize("bad", ["wrong", None, ""])
def test_invalid_or_missing_key_is_401(monkeypatch, bad):
    monkeypatch.setattr(security, "VALID_API_KEYS", ["secret"])
    with pytest.raises(HTTPException) as exc:
        security.get_api_key(bad)
    assert exc.value.status_code == 401
