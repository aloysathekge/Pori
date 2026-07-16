"""Internal model tools for isolated Surface builds and preview inspection."""

from __future__ import annotations

from typing import Any

from ..surface_builds import (
    SurfaceBuildHandler,
    SurfaceBuildParams,
    SurfacePreviewParams,
)

SURFACE_BUILD_CONTEXT_KEY = "surface_builds"
SURFACE_BUILD_TOOL_NAMES = frozenset({"surface_build", "surface_preview"})


def _handler(context: dict) -> SurfaceBuildHandler:
    handler = context.get(SURFACE_BUILD_CONTEXT_KEY)
    if not isinstance(handler, SurfaceBuildHandler):
        raise ValueError("Surface builds are unavailable for this run")
    return handler


async def surface_build_tool(
    params: SurfaceBuildParams,
    context: dict,
) -> dict[str, Any]:
    return await _handler(context).build(params)


async def surface_preview_tool(
    params: SurfacePreviewParams,
    context: dict,
) -> dict[str, Any]:
    return await _handler(context).preview(params)


def register_surface_build_tools(registry) -> None:
    if "surface_build" not in registry.tools:
        registry.register_tool(
            name="surface_build",
            param_model=SurfaceBuildParams,
            function=surface_build_tool,
            description=(
                "Validate and compile one immutable Surface revision using the "
                "fixed Aloy toolchain in an isolated build provider."
            ),
        )
    if "surface_preview" not in registry.tools:
        registry.register_tool(
            name="surface_preview",
            param_model=SurfacePreviewParams,
            function=surface_preview_tool,
            description=(
                "Inspect retained diagnostics and preview artifact metadata for "
                "a Surface build. This does not execute the generated app."
            ),
        )


__all__ = [
    "SURFACE_BUILD_CONTEXT_KEY",
    "SURFACE_BUILD_TOOL_NAMES",
    "register_surface_build_tools",
    "surface_build_tool",
    "surface_preview_tool",
]
