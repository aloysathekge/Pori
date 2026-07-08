"""Google Calendar tools — list + create events — proving the ProviderSpec
abstraction: a second Google capability, same connect-engine, same injected
token, no new OAuth code."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

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
        "id": created.get("id"),
        "html_link": created.get("htmlLink"),
    }
