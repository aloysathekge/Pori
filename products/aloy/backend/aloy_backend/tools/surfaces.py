"""Internal model tools for revision-safe Event Surface source authoring."""

from __future__ import annotations

from typing import Any

from ..surface_authoring import (
    SurfaceAuthoringHandler,
    SurfaceReadProjectParams,
    SurfaceWriteFilesParams,
)

SURFACE_AUTHORING_CONTEXT_KEY = "surface_authoring"
SURFACE_AUTHORING_TOOL_NAMES = frozenset(
    {"surface_read_project", "surface_write_files"}
)


def _handler(context: dict) -> SurfaceAuthoringHandler:
    handler = context.get(SURFACE_AUTHORING_CONTEXT_KEY)
    if not isinstance(handler, SurfaceAuthoringHandler):
        raise ValueError("Surface authoring is unavailable for this run")
    return handler


async def surface_read_project_tool(
    params: SurfaceReadProjectParams,
    context: dict,
) -> dict[str, Any]:
    del params
    return await _handler(context).read()


async def surface_write_files_tool(
    params: SurfaceWriteFilesParams,
    context: dict,
) -> dict[str, Any]:
    return await _handler(context).write(params)


def register_surface_authoring_tools(registry) -> None:
    if "surface_read_project" not in registry.tools:
        registry.register_tool(
            name="surface_read_project",
            param_model=SurfaceReadProjectParams,
            function=surface_read_project_tool,
            description=(
                "Read the current Event Surface project, immutable draft source "
                "snapshot, and the expected revision for the next mutation."
            ),
        )
    if "surface_write_files" not in registry.tools:
        registry.register_tool(
            name="surface_write_files",
            param_model=SurfaceWriteFilesParams,
            function=surface_write_files_tool,
            description=(
                "Atomically persist source patches as a new immutable Surface "
                "draft revision. Requires the revision returned by "
                "surface_read_project and a unique idempotency key. This is the "
                "only tool that makes generated source part of the durable draft; "
                "ordinary filesystem writes do not."
            ),
        )


__all__ = [
    "SURFACE_AUTHORING_CONTEXT_KEY",
    "SURFACE_AUTHORING_TOOL_NAMES",
    "register_surface_authoring_tools",
    "surface_read_project_tool",
    "surface_write_files_tool",
]
