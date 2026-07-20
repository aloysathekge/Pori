"""Google Calendar tools — list + create events — proving the ProviderSpec
abstraction: a second Google capability, same connect-engine, same injected
token, no new OAuth code."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

from pori.tools.registry import ReconciliationStatus, ToolReconciliation

from . import google_common as g

CAL_API = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
CALENDAR_TOOL_NAMES = frozenset({"calendar_list_events", "calendar_create_event"})
CALENDAR_WRITE_TOOLS = frozenset({"calendar_create_event"})


class CalendarListParams(BaseModel):
    time_min: str | None = Field(
        default=None,
        description="ISO-8601 lower bound (RFC3339), e.g. 2026-07-08T00:00:00Z",
    )
    max_results: int = Field(10, ge=1, le=25, description="Max events (1-25)")


def calendar_list_events_tool(
    params: CalendarListParams, context: Dict[str, Any]
) -> Dict[str, Any]:
    query = {
        "maxResults": params.max_results,
        "singleEvents": "true",
        "orderBy": "startTime",
    }
    if params.time_min:
        query["timeMin"] = params.time_min
    try:
        data = g.get(context, CAL_API, query)
    except PermissionError:
        return g.NOT_CONNECTED
    except Exception as exc:
        return {"error": f"Calendar list failed: {exc}"}
    events = []
    for e in data.get("items", []) or []:
        events.append(
            {
                "id": e.get("id"),
                "summary": e.get("summary", ""),
                "start": (e.get("start") or {}).get("dateTime")
                or (e.get("start") or {}).get("date"),
                "end": (e.get("end") or {}).get("dateTime")
                or (e.get("end") or {}).get("date"),
                "location": e.get("location", ""),
            }
        )
    return {"events": events, "count": len(events)}


class CalendarCreateParams(BaseModel):
    summary: str = Field(..., description="Event title")
    start: str = Field(
        ..., description="Start ISO-8601 datetime, e.g. 2026-07-09T15:00:00Z"
    )
    end: str = Field(..., description="End ISO-8601 datetime")
    description: str | None = Field(default=None, description="Optional details")


def calendar_create_event_tool(
    params: CalendarCreateParams, context: Dict[str, Any]
) -> Dict[str, Any]:
    body = {
        "summary": params.summary,
        "start": {"dateTime": params.start},
        "end": {"dateTime": params.end},
    }
    correlation_key = g.execution_correlation_key(
        context, namespace="calendar_create_event"
    )
    if correlation_key:
        # Google explicitly supports caller-chosen base32hex event IDs to
        # prevent duplicates after ambiguous insert failures. A SHA-256 hex
        # digest is a valid subset of that alphabet.
        body["id"] = f"a10{correlation_key}"
    if params.description:
        body["description"] = params.description
    try:
        created = g.post(context, CAL_API, body)
    except PermissionError:
        return g.NOT_CONNECTED
    except Exception as exc:
        return {"error": f"Calendar create failed: {exc}"}
    return {
        "created": True,
        "id": created.get("id") or body.get("id"),
        "html_link": created.get("htmlLink"),
    }


def reconcile_calendar_create_event_tool(
    params: CalendarCreateParams, context: Dict[str, Any]
) -> ToolReconciliation:
    """Read the deterministic provider event ID; never repeat the insert."""
    del params
    correlation_key = g.execution_correlation_key(
        context, namespace="calendar_create_event"
    )
    if not correlation_key:
        return ToolReconciliation(
            status=ReconciliationStatus.UNKNOWN,
            error="The original execution attempt has no provider correlation key.",
        )
    provider_id = f"a10{correlation_key}"
    try:
        event = g.get(context, f"{CAL_API}/{provider_id}")
    except PermissionError:
        return ToolReconciliation(
            status=ReconciliationStatus.UNKNOWN,
            error="Google is not connected, so the event cannot be reconciled.",
        )
    except Exception as exc:
        # A 404 can be eventual visibility or a provider rejection whose local
        # response was lost. Neither proves it is safe to insert again.
        return ToolReconciliation(
            status=ReconciliationStatus.UNKNOWN,
            error=f"Calendar reconciliation failed: {exc}",
        )
    resolved_id = str(event.get("id") or provider_id)
    return ToolReconciliation(
        status=ReconciliationStatus.SUCCEEDED,
        provider_operation_id=resolved_id,
        result={
            "created": True,
            "id": resolved_id,
            "html_link": event.get("htmlLink"),
            "reconciled": True,
        },
        evidence=(
            {
                "provider": "google:calendar",
                "lookup": "deterministic_event_id",
                "provider_operation_id": resolved_id,
            },
        ),
    )
