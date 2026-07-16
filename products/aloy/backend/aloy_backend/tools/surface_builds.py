"""Internal model tools for isolated Surface builds and preview inspection."""

from __future__ import annotations

from typing import Any

from ..surface_builds import (
    SurfaceBuildHandler,
    SurfaceBuildParams,
    SurfacePreviewParams,
    SurfacePublicationParams,
)

SURFACE_BUILD_CONTEXT_KEY = "surface_builds"
SURFACE_BUILD_TOOL_NAMES = frozenset(
    {"surface_build", "surface_preview", "surface_publish", "surface_rollback"}
)


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


async def surface_publish_tool(
    params: SurfacePublicationParams,
    context: dict,
) -> dict[str, Any]:
    return await _handler(context).publish(params)


async def surface_rollback_tool(
    params: SurfacePublicationParams,
    context: dict,
) -> dict[str, Any]:
    return await _handler(context).rollback(params)


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
    if "surface_publish" not in registry.tools:
        registry.register_tool(
            name="surface_publish",
            param_model=SurfacePublicationParams,
            function=surface_publish_tool,
            description=(
                "Atomically publish one successful validated Surface build as "
                "the Event's live last-good revision. Requires the current "
                "published pointers and a unique idempotency key."
            ),
        )
    if "surface_rollback" not in registry.tools:
        registry.register_tool(
            name="surface_rollback",
            param_model=SurfacePublicationParams,
            function=surface_rollback_tool,
            description=(
                "Restore a previously published last-good Surface build without "
                "changing Event data. Requires the current published pointers."
            ),
        )


__all__ = [
    "SURFACE_BUILD_CONTEXT_KEY",
    "SURFACE_BUILD_TOOL_NAMES",
    "register_surface_build_tools",
    "surface_build_tool",
    "surface_publish_tool",
    "surface_preview_tool",
    "surface_rollback_tool",
]
