"""Event-scoped detailed reads for canonical Surface state."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pori import RunContext

from ..database import async_session
from ..surface_state import read_surface_state

SURFACE_STATE_CONTEXT_KEY = "surface_state_reader"
SURFACE_STATE_READ_TOOL_NAME = "surface_state_read"


class SurfaceStateReadParams(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    namespace: str | None = Field(default=None, min_length=1, max_length=64)
    keys: list[str] = Field(default_factory=list, max_length=100)
    limit: int = Field(default=100, ge=1, le=500)


class SurfaceStateReader:
    def __init__(
        self,
        *,
        run_context: RunContext,
        session_factory: Any = async_session,
        owner_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._run_context = run_context
        self._session_factory = session_factory
        self._owner_loop = owner_loop

    async def _read(self, params: SurfaceStateReadParams) -> dict[str, Any]:
        if not self._run_context.event_id:
            raise ValueError("Surface state requires an Event-scoped Run")
        async with self._session_factory() as session:
            return await read_surface_state(
                session,
                organization_id=self._run_context.organization_id,
                user_id=self._run_context.user_id,
                event_id=self._run_context.event_id,
                namespace=params.namespace,
                keys=params.keys,
                limit=params.limit,
            )

    async def read(self, params: SurfaceStateReadParams) -> dict[str, Any]:
        current = asyncio.get_running_loop()
        if self._owner_loop is None or self._owner_loop is current:
            return await self._read(params)
        future = asyncio.run_coroutine_threadsafe(self._read(params), self._owner_loop)
        return await asyncio.wrap_future(future)


async def surface_state_read_tool(
    params: SurfaceStateReadParams, context: dict
) -> dict[str, Any]:
    reader = context.get(SURFACE_STATE_CONTEXT_KEY)
    if not isinstance(reader, SurfaceStateReader):
        raise ValueError("Surface state is unavailable for this Run")
    return await reader.read(params)


def register_surface_state_tools(registry) -> None:
    if SURFACE_STATE_READ_TOOL_NAME in registry.tools:
        return
    registry.register_tool(
        name=SURFACE_STATE_READ_TOOL_NAME,
        param_model=SurfaceStateReadParams,
        function=surface_state_read_tool,
        description=(
            "Read detailed canonical state owned by the current Event Surface. "
            "Use this when the bounded Event context projection is insufficient. "
            "Reads are tenant- and Event-scoped and never inspect iframe state."
        ),
    )


__all__ = [
    "SURFACE_STATE_CONTEXT_KEY",
    "SURFACE_STATE_READ_TOOL_NAME",
    "SurfaceStateReadParams",
    "SurfaceStateReader",
    "register_surface_state_tools",
    "surface_state_read_tool",
]
