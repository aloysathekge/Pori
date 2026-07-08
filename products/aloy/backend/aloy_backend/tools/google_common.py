"""Shared plumbing for Google tools (Gmail, Calendar).

Sync HTTP with the token the connect-engine injected into the run context.
"""

from __future__ import annotations

from typing import Any, Dict, List

import httpx

NOT_CONNECTED = {
    "error": "Google is not connected. Ask the user to connect Google in "
    "Settings → Connections, then try again.",
}


def token(context: Dict[str, Any]) -> str | None:
    return (context or {}).get("connections", {}).get("google", {}).get("access_token")


def _auth(context: Dict[str, Any]) -> dict:
    tok = token(context)
    if not tok:
        raise PermissionError("not_connected")
    return {"Authorization": f"Bearer {tok}"}


def get(context: Dict[str, Any], url: str, params: dict | None = None) -> dict:
    with httpx.Client(timeout=30) as client:
        resp = client.get(url, headers=_auth(context), params=params or {})
        resp.raise_for_status()
        return resp.json()


def post(context: Dict[str, Any], url: str, json_body: dict) -> dict:
    with httpx.Client(timeout=30) as client:
        resp = client.post(url, headers=_auth(context), json=json_body)
        resp.raise_for_status()
        return resp.json()


def header(headers: List[dict], name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""
