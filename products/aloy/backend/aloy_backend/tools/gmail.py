"""Gmail tools — search, read, send — on the user's connected Google account."""

from __future__ import annotations

import base64
from email.message import EmailMessage
from typing import Any, Dict

from pydantic import BaseModel, Field

from . import google_common as g

GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"
GMAIL_TOOL_NAMES = frozenset(
    {
        "gmail_search",
        "gmail_read",
        "gmail_list_drafts",
        "gmail_create_draft",
        "gmail_send_draft",
        "gmail_send",
    }
)
# Consequential write tools — a deployment can HITL-gate these by name
# (hitl.interrupt_on). These DELIVER an email: gmail_send composes+sends a new
# message; gmail_send_draft sends an existing draft. Creating/listing drafts is
# deliberately NOT here: staging a draft has no external consequence, so it
# needs no gate (the Proposal pattern — the agent drafts, the human commits).
GMAIL_WRITE_TOOLS = frozenset({"gmail_send", "gmail_send_draft"})


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


class GmailDraftParams(BaseModel):
    body: str = Field(..., description="Plain-text email body")
    to: str | None = Field(
        default=None, description="Recipient email address (optional for a draft)"
    )
    subject: str = Field("", description="Email subject")
    cc: str | None = Field(default=None, description="Optional CC address")


def gmail_create_draft_tool(
    params: GmailDraftParams, context: Dict[str, Any]
) -> Dict[str, Any]:
    """Save an email to Gmail Drafts WITHOUT sending it, for the user to review
    and send themselves. Prefer this over gmail_send when the user hasn't
    explicitly asked to send, or when the message needs their eyes first."""
    msg = EmailMessage()
    if params.to:
        msg["To"] = params.to
    msg["Subject"] = params.subject
    if params.cc:
        msg["Cc"] = params.cc
    msg.set_content(params.body)
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    try:
        draft = g.post(context, f"{GMAIL_API}/drafts", {"message": {"raw": raw}})
    except PermissionError:
        return g.NOT_CONNECTED
    except Exception as exc:
        return {"error": f"Gmail draft failed: {exc}"}
    return {
        "drafted": True,
        "draft_id": draft.get("id"),
        "message_id": (draft.get("message") or {}).get("id"),
        "to": params.to,
        "note": "Saved to Gmail Drafts — NOT sent. The user reviews and sends it "
        "from Gmail themselves.",
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


class GmailListDraftsParams(BaseModel):
    max_results: int = Field(10, ge=1, le=25, description="Max drafts (1-25)")


def gmail_list_drafts_tool(
    params: GmailListDraftsParams, context: Dict[str, Any]
) -> Dict[str, Any]:
    """List the user's Gmail drafts (id + recipient/subject) so a draft can be
    reviewed or sent by id with gmail_send_draft."""
    try:
        listing = g.get(
            context, f"{GMAIL_API}/drafts", {"maxResults": params.max_results}
        )
    except PermissionError:
        return g.NOT_CONNECTED
    except Exception as exc:
        return {"error": f"Gmail list drafts failed: {exc}"}
    drafts = []
    for d in listing.get("drafts", []) or []:
        draft_id = d.get("id")
        try:
            full = g.get(
                context,
                f"{GMAIL_API}/drafts/{draft_id}",
                {"format": "metadata", "metadataHeaders": ["To", "Subject"]},
            )
        except Exception:
            drafts.append({"draft_id": draft_id})
            continue
        msg = full.get("message", {})
        headers = msg.get("payload", {}).get("headers", [])
        drafts.append(
            {
                "draft_id": draft_id,
                "to": g.header(headers, "To"),
                "subject": g.header(headers, "Subject"),
                "snippet": msg.get("snippet", ""),
            }
        )
    return {"drafts": drafts, "count": len(drafts)}


def gmail_draft_preview(context: Dict[str, Any], draft_id: str) -> Dict[str, str]:
    """Fetch a draft's recipient/subject/body — for the approval card on a
    "send this draft" gate, where the tool call only carries the draft id.
    Best-effort: returns {} if the draft can't be read (the gate still works,
    just with less detail)."""
    try:
        full = g.get(context, f"{GMAIL_API}/drafts/{draft_id}", {"format": "full"})
    except Exception:
        return {}
    payload = full.get("message", {}).get("payload", {})
    headers = payload.get("headers", [])
    preview = {
        "to": g.header(headers, "To"),
        "subject": g.header(headers, "Subject"),
        "body": _decode_body(payload)[:5000],
    }
    return {k: v for k, v in preview.items() if v}


class GmailSendDraftParams(BaseModel):
    draft_id: str = Field(
        ..., description="Draft id (from gmail_create_draft or gmail_list_drafts)"
    )


def gmail_send_draft_tool(
    params: GmailSendDraftParams, context: Dict[str, Any]
) -> Dict[str, Any]:
    """Send an EXISTING draft by id — Gmail removes it from Drafts on send.
    Use this to deliver a draft the user reviewed, instead of composing a new
    message with gmail_send (which would leave the draft behind as a duplicate)."""
    try:
        sent = g.post(context, f"{GMAIL_API}/drafts/send", {"id": params.draft_id})
    except PermissionError:
        return g.NOT_CONNECTED
    except Exception as exc:
        return {"error": f"Gmail send draft failed: {exc}"}
    return {"sent": True, "id": sent.get("id"), "draft_id": params.draft_id}
