"""Assemble one Surface Builder run's scoped virtual workspace."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from pori import (
    CompositeFileBackend,
    FileMount,
    MemoryFileBackend,
    ReadOnlyFileBackend,
    RunContext,
)

from .database import async_session
from .event_presenters import (
    event_payload,
    file_payload,
    proposal_payload,
    task_payload,
    trail_payload,
)
from .models import ActionProposal, Event, EventTrailEntry, StoredFile, Task
from .storage import ObjectStore
from .surface_authoring import SurfaceAuthoringHandler, surface_project_snapshot
from .surface_build_runner import SurfaceBuildRunner
from .surface_builds import SurfaceBuildHandler
from .tools.surface_builds import SURFACE_BUILD_CONTEXT_KEY
from .tools.surfaces import SURFACE_AUTHORING_CONTEXT_KEY


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _json_file(value: Any) -> str:
    return json.dumps(value, indent=2, default=_json_default, sort_keys=True) + "\n"


@dataclass(frozen=True)
class SurfaceAuthoringRuntime:
    """File and tool capabilities granted to one Surface Builder run."""

    file_backend: CompositeFileBackend
    authoring_handler: SurfaceAuthoringHandler
    build_handler: SurfaceBuildHandler
    project_snapshot: dict[str, Any]

    @property
    def tool_context_extra(self) -> dict[str, Any]:
        return {
            SURFACE_AUTHORING_CONTEXT_KEY: self.authoring_handler,
            SURFACE_BUILD_CONTEXT_KEY: self.build_handler,
        }


async def resolve_surface_authoring_runtime(
    session: AsyncSession,
    *,
    run_context: RunContext,
    session_factory: Any = async_session,
    owner_loop: asyncio.AbstractEventLoop | None = None,
    build_runner: SurfaceBuildRunner | None = None,
    object_store: ObjectStore | None = None,
) -> SurfaceAuthoringRuntime:
    """Project canonical Event truth and the current draft into virtual mounts."""
    event_id = run_context.event_id
    if not event_id:
        raise ValueError("Event identity is required for Surface authoring")
    event = await session.get(Event, event_id)
    if (
        event is None
        or event.organization_id != run_context.organization_id
        or event.user_id != run_context.user_id
    ):
        raise ValueError("Event is unavailable")
    if event.lifecycle == "archived":
        raise ValueError("Event is archived")

    tasks = list(
        (
            await session.execute(
                select(Task)
                .where(
                    col(Task.organization_id) == event.organization_id,
                    col(Task.user_id) == event.user_id,
                    col(Task.event_id) == event.id,
                )
                .order_by(col(Task.order), col(Task.created_at))
                .limit(200)
            )
        )
        .scalars()
        .all()
    )
    proposals = list(
        (
            await session.execute(
                select(ActionProposal)
                .where(
                    col(ActionProposal.organization_id) == event.organization_id,
                    col(ActionProposal.user_id) == event.user_id,
                    col(ActionProposal.event_id) == event.id,
                )
                .order_by(col(ActionProposal.created_at).desc())
                .limit(100)
            )
        )
        .scalars()
        .all()
    )
    files = list(
        (
            await session.execute(
                select(StoredFile)
                .where(
                    col(StoredFile.organization_id) == event.organization_id,
                    col(StoredFile.user_id) == event.user_id,
                    col(StoredFile.event_id) == event.id,
                )
                .order_by(col(StoredFile.created_at).desc())
                .limit(200)
            )
        )
        .scalars()
        .all()
    )
    trail = list(
        (
            await session.execute(
                select(EventTrailEntry)
                .where(
                    col(EventTrailEntry.organization_id) == event.organization_id,
                    col(EventTrailEntry.user_id) == event.user_id,
                    col(EventTrailEntry.event_id) == event.id,
                )
                .order_by(col(EventTrailEntry.created_at).desc())
                .limit(200)
            )
        )
        .scalars()
        .all()
    )
    project = await surface_project_snapshot(
        session,
        organization_id=event.organization_id,
        user_id=event.user_id,
        event_id=event.id,
        include_files=True,
    )

    event_files = {
        "/README.md": (
            "# Event context\n\n"
            "This mount is a read-only projection of canonical Aloy data. "
            "Editing generated source never changes these records.\n"
        ),
        "/event.json": _json_file(event_payload(event)),
        "/tasks.json": _json_file([task_payload(task) for task in tasks]),
        "/proposals.json": _json_file(
            [proposal_payload(proposal) for proposal in proposals]
        ),
        "/files.json": _json_file([file_payload(file) for file in files]),
        "/trail.json": _json_file([trail_payload(entry) for entry in trail]),
    }
    draft = project.get("draft") or {}
    workspace_files = {
        str(path): str(content)
        for path, content in dict(draft.get("files") or {}).items()
    }
    event_backend = ReadOnlyFileBackend(MemoryFileBackend(event_files))
    workspace_backend = MemoryFileBackend(workspace_files)
    file_backend = CompositeFileBackend(
        (
            FileMount("/event", event_backend, read_only=True),
            FileMount("/workspace", workspace_backend),
        )
    )
    return SurfaceAuthoringRuntime(
        file_backend=file_backend,
        authoring_handler=SurfaceAuthoringHandler(
            run_context=run_context,
            session_factory=session_factory,
            owner_loop=owner_loop,
        ),
        build_handler=SurfaceBuildHandler(
            run_context=run_context,
            runner=build_runner,
            object_store=object_store,
            session_factory=session_factory,
            owner_loop=owner_loop,
        ),
        project_snapshot=project,
    )


__all__ = ["SurfaceAuthoringRuntime", "resolve_surface_authoring_runtime"]
