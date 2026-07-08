"""Gmail tools (Aloy product tools) — act on the user's connected account.

The user's fresh access token is resolved by the connect-engine at run-setup
and injected into the tool context under ``connections.google``; these sync
tools read it and call the Gmail REST API. No token = a friendly "connect
Gmail" message (the tools are also excluded from the surface when unconnected,
so this is a belt-and-braces fallback).
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List

import httpx
from pori.tools.registry import CapabilityGroup
from pydantic import BaseModel, Field

GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"
GMAIL_TOOL_NAMES = frozenset({"gmail_search", "gmail_read"})

_NOT_CONNECTED = {
    "error": "Gmail is not connected. Ask the user to connect Gmail in "
    "Settings → Connections, then try again.",
}


def _token(context: Dict[str, Any]) -> str | None:
    return (context or {}).get("connections", {}).get("google", {}).get("access_token")


def _get(context: Dict[str, Any], path: str, params: dict | None = None) -> dict:
    token = _token(context)
    if not token:
        raise PermissionError("not_connected")
    with httpx.Client(timeout=30) as client:
        resp = client.get(
            f"{GMAIL_API}{path}",
            headers={"Authorization": f"Bearer {token}"},
            params=params or {},
        )
        resp.raise_for_status()
        return resp.json()


def _header(headers: List[dict], name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _decode_body(payload: dict) -> str:
    """Best-effort plain-text body from a Gmail message payload."""
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data")
        if data:
            return base64.urlsafe_b64decode(data + "===").decode("utf-8", "replace")
    for part in payload.get("parts", []) or []:
        text = _decode_body(part)
        if text:
            return text
    return ""


class GmailSearchParams(BaseModel):
    query: str = Field(
        ..., description="Gmail search query, e.g. 'from:alice is:unread newer_than:7d'"
    )
    max_results: int = Field(
        10, ge=1, le=25, description="Max messages to return (1-25)"
    )


def gmail_search_tool(
    params: GmailSearchParams, context: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        listing = _get(
            context,
            "/messages",
            {"q": params.query, "maxResults": params.max_results},
        )
    except PermissionError:
        return _NOT_CONNECTED
    except Exception as exc:
        return {"error": f"Gmail search failed: {exc}"}
    results = []
    for m in listing.get("messages", []) or []:
        try:
            msg = _get(
                context,
                f"/messages/{m['id']}",
                {
                    "format": "metadata",
                    "metadataHeaders": ["From", "Subject", "Date"],
                },
            )
        except Exception:
            continue
        headers = msg.get("payload", {}).get("headers", [])
        results.append(
            {
                "id": m["id"],
                "from": _header(headers, "From"),
                "subject": _header(headers, "Subject"),
                "date": _header(headers, "Date"),
                "snippet": msg.get("snippet", ""),
            }
        )
    return {"messages": results, "count": len(results)}


class GmailReadParams(BaseModel):
    message_id: str = Field(..., description="The Gmail message id (from gmail_search)")


def gmail_read_tool(params: GmailReadParams, context: Dict[str, Any]) -> Dict[str, Any]:
    try:
        msg = _get(context, f"/messages/{params.message_id}", {"format": "full"})
    except PermissionError:
        return _NOT_CONNECTED
    except Exception as exc:
        return {"error": f"Gmail read failed: {exc}"}
    payload = msg.get("payload", {})
    headers = payload.get("headers", [])
    return {
        "id": params.message_id,
        "from": _header(headers, "From"),
        "to": _header(headers, "To"),
        "subject": _header(headers, "Subject"),
        "date": _header(headers, "Date"),
        "body": _decode_body(payload)[:20000],
    }


def register_gmail_tools(registry) -> None:
    """Register the Gmail tools + the 'google' capability group (idempotent)."""
    if "gmail_search" in registry.tools:
        return
    registry.register_tool(
        name="gmail_search",
        param_model=GmailSearchParams,
        function=gmail_search_tool,
        description=(
            "Search the user's connected Gmail (queries use Gmail search syntax) "
            "and return matching message summaries."
        ),
    )
    registry.register_tool(
        name="gmail_read",
        param_model=GmailReadParams,
        function=gmail_read_tool,
        description="Read the full text of one Gmail message by id.",
    )
    try:
        registry.define_group(
            CapabilityGroup(
                name="google",
                description="Act on the user's connected Google account (Gmail).",
                tool_names=GMAIL_TOOL_NAMES,
            )
        )
    except ValueError:
        pass  # already defined
