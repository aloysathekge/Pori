"""Conversation-facing control-plane tool for model-planned Surfaces."""

from __future__ import annotations

from typing import Any

from ..surface_requests import SurfaceRequestHandler, SurfaceRequestParams

SURFACE_REQUEST_CONTEXT_KEY = "surface_requests"
SURFACE_REQUEST_TOOL_NAME = "request_event_surface"


def _handler(context: dict) -> SurfaceRequestHandler:
    handler = context.get(SURFACE_REQUEST_CONTEXT_KEY)
    if not isinstance(handler, SurfaceRequestHandler):
        raise ValueError("Surface requests are unavailable for this Run")
    return handler


async def request_event_surface_tool(
    params: SurfaceRequestParams,
    context: dict,
) -> dict[str, Any]:
    return await _handler(context).request(params)


def register_surface_request_tool(registry) -> None:
    if SURFACE_REQUEST_TOOL_NAME in registry.tools:
        return
    registry.register_tool(
        name=SURFACE_REQUEST_TOOL_NAME,
        param_model=SurfaceRequestParams,
        function=request_event_surface_tool,
        description=(
            "Request creation or revision of the current Event's durable, visual, "
            "interactive Surface when that experience is more useful than a chat "
            "answer, document, or collection of Tasks. Choose this from the meaning "
            "and long-term user value of the request: recurring structured views "
            "such as timetables, plans, maps, dashboards, trackers, and multi-view "
            "workspaces are strong candidates. This queues a separate specialized "
            "builder; it does not mean the Surface is ready. Never claim that a "
            "Surface exists, is live, or was published from this tool's queued result."
        ),
    )


__all__ = [
    "SURFACE_REQUEST_CONTEXT_KEY",
    "SURFACE_REQUEST_TOOL_NAME",
    "register_surface_request_tool",
    "request_event_surface_tool",
]
