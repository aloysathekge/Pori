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
from .models import (
    ActionProposal,
    Event,
    EventBrief,
    EventTrailEntry,
    StoredFile,
    Task,
)
from .storage import ObjectStore, get_object_store, safe_name
from .surface_authoring import SurfaceAuthoringHandler, surface_project_snapshot
from .surface_build_runner import (
    SurfaceBuildRunner,
    configured_surface_build_runner,
)
from .surface_builds import SurfaceBuildHandler
from .tools.surface_builds import SURFACE_BUILD_CONTEXT_KEY
from .tools.surfaces import SURFACE_AUTHORING_CONTEXT_KEY


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _json_file(value: Any) -> str:
    return json.dumps(value, indent=2, default=_json_default, sort_keys=True) + "\n"


_MAX_CONTEXT_FILE_BYTES = 256 * 1024
_MAX_CONTEXT_FILES_BYTES = 1024 * 1024
_TEXT_EXTENSIONS = frozenset(
    {".csv", ".html", ".json", ".md", ".rst", ".text", ".tsv", ".txt", ".xml"}
)


def _is_text_context_file(file: StoredFile) -> bool:
    content_type = (file.content_type or "").lower()
    suffix = "." + file.name.rsplit(".", 1)[-1].lower() if "." in file.name else ""
    return (
        content_type.startswith("text/")
        or content_type in {"application/json", "application/xml"}
        or suffix in _TEXT_EXTENSIONS
    )


async def _read_context_file(store: ObjectStore, file: StoredFile) -> str | None:
    def _read() -> bytes:
        with store.open(file.storage_key) as stream:
            return stream.read(_MAX_CONTEXT_FILE_BYTES + 1)

    try:
        content = await asyncio.to_thread(_read)
    except (FileNotFoundError, OSError):
        return None
    if len(content) > _MAX_CONTEXT_FILE_BYTES:
        return None
    return content.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class SurfaceAuthoringRuntime:
    """File and tool capabilities granted to one Surface Builder run."""

    file_backend: CompositeFileBackend
    authoring_handler: SurfaceAuthoringHandler
    build_handler: SurfaceBuildHandler
    workspace_build_runner: SurfaceBuildRunner
    project_snapshot: dict[str, Any]
    prompt_context: dict[str, Any]

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
    brief = (
        (
            await session.execute(
                select(EventBrief)
                .where(
                    col(EventBrief.organization_id) == event.organization_id,
                    col(EventBrief.user_id) == event.user_id,
                    col(EventBrief.event_id) == event.id,
                    col(EventBrief.status) == "active",
                )
                .order_by(col(EventBrief.version).desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    project = await surface_project_snapshot(
        session,
        organization_id=event.organization_id,
        user_id=event.user_id,
        event_id=event.id,
        include_files=True,
    )

    file_manifest: list[dict[str, Any]] = []
    mounted_files: dict[str, str] = {}
    mounted_bytes = 0
    store: ObjectStore | None = object_store
    for file in files:
        item = file_payload(file)
        if (
            _is_text_context_file(file)
            and file.size_bytes <= _MAX_CONTEXT_FILE_BYTES
            and mounted_bytes < _MAX_CONTEXT_FILES_BYTES
        ):
            store = store or get_object_store()
            content = await _read_context_file(store, file)
            encoded_size = len(content.encode("utf-8")) if content is not None else 0
            if (
                content is not None
                and mounted_bytes + encoded_size <= _MAX_CONTEXT_FILES_BYTES
            ):
                workspace_path = f"/files/{file.id}/{safe_name(file.name)}"
                mounted_files[workspace_path] = content
                mounted_bytes += encoded_size
                item["workspace_path"] = f"/event{workspace_path}"
        file_manifest.append(item)

    event_files = {
        "/README.md": (
            "# Event context\n\n"
            "This mount is a read-only projection of canonical Aloy data. "
            "Editing generated source never changes these records. Text files "
            "listed with a workspace_path are available below /event/files.\n"
        ),
        "/event.json": _json_file(event_payload(event)),
        "/brief.json": _json_file(brief.payload if brief is not None else {}),
        "/tasks.json": _json_file([task_payload(task) for task in tasks]),
        "/proposals.json": _json_file(
            [proposal_payload(proposal) for proposal in proposals]
        ),
        "/files.json": _json_file(file_manifest),
        "/trail.json": _json_file([trail_payload(entry) for entry in trail]),
        **mounted_files,
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
    resolved_build_runner = build_runner or configured_surface_build_runner()
    return SurfaceAuthoringRuntime(
        file_backend=file_backend,
        authoring_handler=SurfaceAuthoringHandler(
            run_context=run_context,
            session_factory=session_factory,
            owner_loop=owner_loop,
        ),
        build_handler=SurfaceBuildHandler(
            run_context=run_context,
            runner=resolved_build_runner,
            object_store=object_store,
            session_factory=session_factory,
            owner_loop=owner_loop,
        ),
        workspace_build_runner=resolved_build_runner,
        project_snapshot=project,
        prompt_context={
            "event": event_payload(event),
            "brief": brief.payload if brief is not None else {},
            "tasks": [task_payload(task) for task in tasks],
            "proposals": [proposal_payload(proposal) for proposal in proposals],
            "files": file_manifest,
            "file_excerpts": mounted_files,
            "trail": [trail_payload(entry) for entry in trail],
            "surface": project,
        },
    )


__all__ = ["SurfaceAuthoringRuntime", "resolve_surface_authoring_runtime"]
