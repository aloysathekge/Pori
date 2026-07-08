"""Aloy product tools — capabilities the PRODUCT adds on top of the kernel.

Registered onto the kernel's ToolRegistry from the Aloy backend; the kernel
stays product-agnostic (one-way dependency rule). All Google tools (Gmail +
Calendar) share one connection and live in the 'google' capability group,
gated per-user by whether the user has connected Google.
"""

from pori.tools.registry import CapabilityGroup

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
    GmailReadParams,
    GmailSearchParams,
    GmailSendParams,
    gmail_read_tool,
    gmail_search_tool,
    gmail_send_tool,
)

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
        "gmail_send",
        GmailSendParams,
        gmail_send_tool,
        "Send an email from the user's connected Gmail.",
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
    "register_google_tools",
]
