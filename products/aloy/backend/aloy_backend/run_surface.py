"""The run surface: ONE place that assembles what a run can reach.

Every agent run — chat (streaming or blocking), durable worker, and later
the Event system's background agents — gets its capability surface from
here: the user's live connections (Gmail/Calendar tokens), their + the
org's MCP servers, the file-library manifest, and the denied-tools set that
gates capability-backed tools off when their backing is absent.

History: this wiring used to live inline in ``send_message`` only; the
durable worker built runs WITHOUT connections/MCP/library (the drift the
2026-07-11 audit flagged). One resolver, no drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from .connections.mcp_store import resolve_run_mcp_servers
from .connections.store import resolve_run_connections
from .library import library_manifest
from .models import Event, StoredFile
from .tenancy import OrganizationPolicy
from .tools import GOOGLE_TOOL_NAMES, LIBRARY_TOOL_NAMES


@dataclass
class RunSurface:
    """Everything a run's orchestration needs about the caller's reach."""

    connections: dict = field(default_factory=dict)
    mcp_servers: list = field(default_factory=list)
    library: list = field(default_factory=list)
    # policy-denied + capability-gated (no google connection / empty library)
    denied_tools: tuple = ()

    @property
    def tool_context_extra(self) -> dict[str, Any]:
        """The per-run tool context this surface contributes."""
        return {"connections": self.connections, "library_files": self.library}


async def resolve_run_surface(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    event_id: str,
    policy: OrganizationPolicy,
    explicit_file_ids: tuple[str, ...] = (),
) -> RunSurface:
    """Resolve capabilities within the owning Event's authority boundary."""
    connections = await resolve_run_connections(session, organization_id, user_id)
    mcp_servers = await resolve_run_mcp_servers(session, organization_id, user_id)
    event = await session.get(Event, event_id)
    retained_scope = col(StoredFile.in_library) == True  # noqa: E712
    if event is None or not event.is_life:
        retained_scope = and_(retained_scope, col(StoredFile.event_id) == event_id)
    file_scope = retained_scope
    if explicit_file_ids:
        file_scope = or_(
            retained_scope,
            and_(
                col(StoredFile.event_id) == event_id,
                col(StoredFile.id).in_(explicit_file_ids),
            ),
        )
    library_statement = select(StoredFile).where(
        col(StoredFile.organization_id) == organization_id,
        col(StoredFile.user_id) == user_id,
        file_scope,
    )
    library_rows = (await session.execute(library_statement)).scalars().all()
    library = library_manifest(list(library_rows))

    connection_denied = () if "google" in connections else tuple(GOOGLE_TOOL_NAMES)
    library_denied = () if library else tuple(LIBRARY_TOOL_NAMES)
    return RunSurface(
        connections=connections,
        mcp_servers=list(mcp_servers),
        library=library,
        denied_tools=tuple(policy.denied_tools) + connection_denied + library_denied,
    )
