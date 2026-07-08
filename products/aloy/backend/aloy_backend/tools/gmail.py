"""Gmail tools — search, read, send — on the user's connected Google account."""

from __future__ import annotations

import base64
from email.message import EmailMessage
from typing import Any, Dict

from pydantic import BaseModel, Field

from . import google_common as g

GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"
GMAIL_TOOL_NAMES = frozenset({"gmail_search", "gmail_read", "gmail_send"})
# Write tools — a deployment can HITL-gate these by name (hitl.interrupt_on).
GMAIL_WRITE_TOOLS = frozenset({"gmail_send"})


def _decode_body(payload: dict) -> str:
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
    max_results: int = Field(10, ge=1, le=25, description="Max messages (1-25)")


def gmail_search_tool(
    params: GmailSearchParams, context: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        listing = g.get(
            context,
            f"{GMAIL_API}/messages",
            {"q": params.query, "maxResults": params.max_results},
        )
    except PermissionError:
        return g.NOT_CONNECTED
    except Exception as exc:
        return {"error": f"Gmail search failed: {exc}"}
    results = []
    for m in listing.get("messages", []) or []:
        try:
            msg = g.get(
                context,
                f"{GMAIL_API}/messages/{m['id']}",
                {"format": "metadata", "metadataHeaders": ["From", "Subject", "Date"]},
            )
        except Exception:
            continue
        headers = msg.get("payload", {}).get("headers", [])
        results.append(
            {
                "id": m["id"],
                "from": g.header(headers, "From"),
                "subject": g.header(headers, "Subject"),
                "date": g.header(headers, "Date"),
                "snippet": msg.get("snippet", ""),
            }
        )
    return {"messages": results, "count": len(results)}


class GmailReadParams(BaseModel):
    message_id: str = Field(..., description="Gmail message id (from gmail_search)")


def gmail_read_tool(params: GmailReadParams, context: Dict[str, Any]) -> Dict[str, Any]:
    try:
        msg = g.get(
            context, f"{GMAIL_API}/messages/{params.message_id}", {"format": "full"}
        )
    except PermissionError:
        return g.NOT_CONNECTED
    except Exception as exc:
        return {"error": f"Gmail read failed: {exc}"}
    payload = msg.get("payload", {})
    headers = payload.get("headers", [])
    return {
        "id": params.message_id,
        "from": g.header(headers, "From"),
        "to": g.header(headers, "To"),
        "subject": g.header(headers, "Subject"),
        "date": g.header(headers, "Date"),
        "body": _decode_body(payload)[:20000],
    }


class GmailSendParams(BaseModel):
    to: str = Field(..., description="Recipient email address")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Plain-text email body")
    cc: str | None = Field(default=None, description="Optional CC address")


def gmail_send_tool(params: GmailSendParams, context: Dict[str, Any]) -> Dict[str, Any]:
    msg = EmailMessage()
    msg["To"] = params.to
    msg["Subject"] = params.subject
    if params.cc:
        msg["Cc"] = params.cc
    msg.set_content(params.body)
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    try:
        sent = g.post(context, f"{GMAIL_API}/messages/send", {"raw": raw})
    except PermissionError:
        return g.NOT_CONNECTED
    except Exception as exc:
        return {"error": f"Gmail send failed: {exc}"}
    return {"sent": True, "id": sent.get("id"), "to": params.to}
