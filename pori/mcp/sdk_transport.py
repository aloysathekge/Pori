"""Default MCP connection backed by the official ``mcp`` SDK (optional extra).

Lazy-imported only when a run actually connects a server without a custom
factory, so importing ``pori`` never requires the ``mcp`` package. HTTP
(Streamable HTTP) and SSE transports; stdio is a later phase.

This is the one part not exercised by the test suite (it needs the SDK + a live
server); it's a thin adapter over the SDK, kept small on purpose.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List

from .config import McpServerConfig
from .session import McpToolDesc


def _import_sdk():
    try:
        from mcp import ClientSession  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "MCP support requires the 'mcp' package: pip install 'pori[mcp]'"
        ) from exc
    return ClientSession


def _http_client():
    # The Streamable HTTP client moved names across SDK versions.
    try:
        from mcp.client.streamable_http import streamablehttp_client  # type: ignore

        return streamablehttp_client
    except ImportError:  # pragma: no cover
        from mcp.client.streamable_http import (
            streamable_http_client as streamablehttp_client,  # type: ignore
        )

        return streamablehttp_client


class _SdkAdapter:
    def __init__(self, session: Any):
        self._session = session

    async def list_tools(self) -> List[McpToolDesc]:
        res = await self._session.list_tools()
        out: List[McpToolDesc] = []
        for tool in res.tools:
            out.append(
                McpToolDesc(
                    name=tool.name,
                    description=getattr(tool, "description", "") or "",
                    input_schema=getattr(tool, "inputSchema", None) or {},
                )
            )
        return out

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        res = await self._session.call_tool(name, arguments=arguments)
        parts: List[str] = []
        for block in getattr(res, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else "(no text content)"


@asynccontextmanager
async def sdk_connection(config: McpServerConfig):
    ClientSession = _import_sdk()
    if config.transport == "sse":
        from mcp.client.sse import sse_client  # type: ignore

        transport = sse_client(config.url, headers=config.headers or None)
    else:
        transport = _http_client()(config.url, headers=config.headers or None)

    async with transport as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield _SdkAdapter(session)
