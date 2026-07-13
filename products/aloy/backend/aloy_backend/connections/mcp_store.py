"""Resolve a run's MCP servers — the same scope/union machinery as connections.

Returns a list of kernel ``McpServerConfig`` (tenancy-blind): the member's own
enabled servers + the org's shared enabled servers for names the member hasn't
personally taken. Auth is resolved to headers here (the kernel never does OAuth).
"""

from __future__ import annotations

import logging
from typing import List

from sqlmodel import select

from pori import McpServerConfig

from ..models import ORG_CONNECTION_USER, McpServer
from .crypto import decrypt

logger = logging.getLogger("aloy_backend")


def _eff_user(scope: str, user_id: str) -> str:
    return ORG_CONNECTION_USER if scope == "org" else user_id


def _headers_for(server: McpServer) -> dict:
    if server.auth_kind == "static" and server.static_secret_enc:
        try:
            return {"Authorization": f"Bearer {decrypt(server.static_secret_enc)}"}
        except Exception:
            logger.warning("Failed to decrypt MCP secret for %s", server.id)
    return {}


def _to_config(server: McpServer) -> McpServerConfig:
    return McpServerConfig(
        name=server.name,
        transport=server.transport if server.transport in ("http", "sse") else "http",
        url=server.url,
        headers=_headers_for(server),
        tools_include=server.tools_include,
        tools_exclude=server.tools_exclude or [],
    )


# Public alias: the test-connection endpoint maps a single row the same way
# runs do — one mapping, no drift.
server_to_config = _to_config


async def resolve_run_mcp_servers(
    session, organization_id: str, user_id: str
) -> List[McpServerConfig]:
    out: List[McpServerConfig] = []
    taken: set[str] = set()

    async def _add(rows) -> None:
        for server in rows:
            if not server.enabled or server.name in taken:
                continue
            taken.add(server.name)
            out.append(_to_config(server))

    # 1) the member's own servers (precedence on name)
    user_rows = (
        (
            await session.execute(
                select(McpServer).where(
                    McpServer.organization_id == organization_id,
                    McpServer.user_id == user_id,
                    McpServer.scope == "user",
                )
            )
        )
        .scalars()
        .all()
    )
    await _add(user_rows)

    # 2) org-shared servers fill names the member hasn't taken
    org_rows = (
        (
            await session.execute(
                select(McpServer).where(
                    McpServer.organization_id == organization_id,
                    McpServer.scope == "org",
                )
            )
        )
        .scalars()
        .all()
    )
    await _add(org_rows)
    return out
