"""Aloy product tools — capabilities the PRODUCT adds on top of the kernel.

Registered onto the kernel's ToolRegistry from the Aloy backend; the kernel
stays product-agnostic (one-way dependency rule). All Google tools (Gmail +
Calendar) share one connection and live in the 'google' capability group,
gated per-user by whether the user has connected Google.
"""

from pori import CapabilityGroup

from .calendar import (
    CALENDAR_TOOL_NAMES,
    CALENDAR_WRITE_TOOLS,
    CalendarCreateParams,
    CalendarListParams,
    calendar_create_event_tool,
    calendar_list_events_tool,
)
from .gmail import (
    GMAIL_TOOL_NAMES,
    GMAIL_WRITE_TOOLS,
    GmailDraftParams,
    GmailListDraftsParams,
    GmailReadParams,
    GmailSearchParams,
    GmailSendDraftParams,
    GmailSendParams,
    gmail_create_draft_tool,
    gmail_draft_preview,
    gmail_list_drafts_tool,
    gmail_read_tool,
    gmail_search_tool,
    gmail_send_draft_tool,
    gmail_send_tool,
)
from .library import LIBRARY_TOOL_NAMES, FetchMyFileParams, fetch_my_file_tool
from .tasks import TaskMutationHandler, register_task_tools

# Every tool the 'google' connection unlocks, and the write subset a deployment
# may want to HITL-gate (hitl.interrupt_on: {gmail_send: true, ...}).
GOOGLE_TOOL_NAMES = GMAIL_TOOL_NAMES | CALENDAR_TOOL_NAMES
GOOGLE_WRITE_TOOLS = GMAIL_WRITE_TOOLS | CALENDAR_WRITE_TOOLS

_TOOLS = [
    (
        "gmail_search",
        GmailSearchParams,
        gmail_search_tool,
        "Search the user's connected Gmail (Gmail query syntax) for messages.",
    ),
    (
        "gmail_read",
        GmailReadParams,
        gmail_read_tool,
        "Read the full text of one Gmail message by id.",
    ),
    (
        "gmail_create_draft",
        GmailDraftParams,
        gmail_create_draft_tool,
        "Save an email to the user's Gmail Drafts WITHOUT sending it, for them "
        "to review and send. Prefer this over gmail_send unless the user "
        "explicitly asked to send. Returns a draft_id.",
    ),
    (
        "gmail_list_drafts",
        GmailListDraftsParams,
        gmail_list_drafts_tool,
        "List the user's Gmail drafts (draft_id, recipient, subject) — use to "
        "find a draft's id before sending it with gmail_send_draft.",
    ),
    (
        "gmail_send_draft",
        GmailSendDraftParams,
        gmail_send_draft_tool,
        "Send an EXISTING draft by draft_id (Gmail removes it from Drafts on "
        "send). Use this to deliver a draft the user reviewed — NOT gmail_send, "
        "which composes a new message and would leave the draft as a duplicate.",
    ),
    (
        "gmail_send",
        GmailSendParams,
        gmail_send_tool,
        "Compose and send a NEW email from the user's connected Gmail. Delivers "
        "immediately. Use gmail_create_draft unless the user asked to send; to "
        "send a draft they already have, use gmail_send_draft.",
    ),
    (
        "calendar_list_events",
        CalendarListParams,
        calendar_list_events_tool,
        "List upcoming events from the user's primary Google Calendar.",
    ),
    (
        "calendar_create_event",
        CalendarCreateParams,
        calendar_create_event_tool,
        "Create an event on the user's primary Google Calendar.",
    ),
]


def register_library_tools(registry) -> None:
    """Register the file-library tool (idempotent). Excluded per-run via
    denied_tools when the user's library is empty — zero context cost until
    it's actually usable (footprint-ladder rung 3)."""
    if "fetch_my_file" in registry.tools:
        return
    registry.register_tool(
        name="fetch_my_file",
        param_model=FetchMyFileParams,
        function=fetch_my_file_tool,
        description=(
            "Retrieve a file from the user's saved file library (their "
            "durable personal files, e.g. a CV) into the workspace so you "
            "can read or edit it."
        ),
    )


def register_google_tools(registry) -> None:
    """Register all Google tools + the 'google' capability group (idempotent)."""
    if "gmail_search" in registry.tools:
        return
    for name, params, fn, desc in _TOOLS:
        registry.register_tool(
            name=name, param_model=params, function=fn, description=desc
        )
    try:
        registry.define_group(
            CapabilityGroup(
                name="google",
                description="Act on the user's connected Google account "
                "(Gmail + Calendar).",
                tool_names=GOOGLE_TOOL_NAMES,
            )
        )
    except ValueError:
        pass  # already defined


__all__ = [
    "GOOGLE_TOOL_NAMES",
    "GOOGLE_WRITE_TOOLS",
    "LIBRARY_TOOL_NAMES",
    "gmail_draft_preview",
    "register_google_tools",
    "register_library_tools",
    "register_task_tools",
    "TaskMutationHandler",
]
