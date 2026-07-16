"""Durable, revision-safe source authoring for model-authored Event Surfaces.

This module owns source persistence only. Building, previewing, publishing,
browser execution, and interaction handling are separate later boundaries.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from pori import RunContext, normalize_virtual_path

from .database import async_session
from .models import Event, EventTrailEntry, SurfaceProject, SurfaceRevision

MAX_SURFACE_FILES = 200
MAX_SURFACE_FILE_BYTES = 256 * 1024
MAX_SURFACE_SOURCE_BYTES = 5 * 1024 * 1024
SURFACE_SDK_VERSION = "1"

_ALLOWED_SOURCE_EXTENSIONS = {
    ".css",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".svg",
    ".ts",
    ".tsx",
}
_FORBIDDEN_SOURCE_NAMES = {
    "bun.lock",
    "bun.lockb",
    "package-lock.json",
    "package.json",
    "pnpm-lock.yaml",
    "tsconfig.json",
    "vite.config.js",
    "vite.config.ts",
    "yarn.lock",
}


class SurfaceAuthoringError(ValueError):
    """Base error returned through the product tool boundary."""


class SurfaceConflictError(SurfaceAuthoringError):
    """The caller authored against a stale or conflicting revision."""


class SurfaceFilePatch(BaseModel):
    path: str = Field(min_length=1, max_length=500)
    operation: Literal["write", "delete"] = "write"
    content: str | None = None

    @model_validator(mode="after")
    def validate_operation(self) -> "SurfaceFilePatch":
        if self.operation == "write" and self.content is None:
            raise ValueError("content is required for a write patch")
        if self.operation == "delete" and self.content is not None:
            raise ValueError("content must be omitted for a delete patch")
        _project_source_path(self.path)
        if self.content is not None:
            size = len(self.content.encode("utf-8"))
            if size > MAX_SURFACE_FILE_BYTES:
                raise ValueError(
                    f"Surface source file exceeds {MAX_SURFACE_FILE_BYTES} bytes"
                )
        return self


class SurfaceWriteFilesParams(BaseModel):
    expected_revision: str | None = None
    idempotency_key: str = Field(min_length=8, max_length=200)
    patches: list[SurfaceFilePatch] = Field(min_length=1, max_length=100)

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("idempotency_key must be trimmed")
        return value

    @model_validator(mode="after")
    def reject_duplicate_paths(self) -> "SurfaceWriteFilesParams":
        paths = [_project_source_path(patch.path) for patch in self.patches]
        if len(paths) != len(set(paths)):
            raise ValueError("Each Surface path may appear only once per mutation")
        return self


class SurfaceReadProjectParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _project_source_path(path: str) -> str:
    """Normalize one model-owned workspace path to its project-relative path."""
    try:
        normalized = normalize_virtual_path(path)
    except ValueError as exc:
        raise ValueError(f"Invalid Surface source path: {exc}") from exc
    if not normalized.startswith("/workspace/"):
        raise ValueError("Surface source paths must live under /workspace/")
    source_path = normalized.removeprefix("/workspace")
    parts = PurePosixPath(source_path).parts
    if any(part.startswith(".") for part in parts if part != "/"):
        raise ValueError("Hidden Surface source paths are not allowed")
    lowered = {part.lower() for part in parts}
    if lowered.intersection({"node_modules", ".git", ".pori"}):
        raise ValueError("Generated dependency and control directories are forbidden")
    name = PurePosixPath(source_path).name.lower()
    if name in _FORBIDDEN_SOURCE_NAMES:
        raise ValueError(f"Aloy owns the generated app toolchain file: {name}")
    if PurePosixPath(source_path).suffix.lower() not in _ALLOWED_SOURCE_EXTENSIONS:
        raise ValueError(f"Unsupported Surface source extension: {source_path}")
    return source_path


def _request_fingerprint(params: SurfaceWriteFilesParams) -> str:
    body = {
        "expected_revision": params.expected_revision,
        "patches": [
            {
                "path": _project_source_path(patch.path),
                "operation": patch.operation,
                "content": patch.content,
            }
            for patch in params.patches
        ],
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_checksum(manifest: dict, files: dict[str, str]) -> str:
    encoded = json.dumps(
        {"manifest": manifest, "files": files},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _revision_payload(
    revision: SurfaceRevision | None,
    *,
    include_files: bool,
) -> dict[str, Any] | None:
    if revision is None:
        return None
    payload: dict[str, Any] = {
        "id": revision.id,
        "revision_number": revision.revision_number,
        "parent_revision_id": revision.parent_revision_id,
        "creator_run_id": revision.creator_run_id,
        "manifest": revision.manifest,
        "checksum": revision.checksum,
        "file_count": revision.file_count,
        "total_bytes": revision.total_bytes,
        "created_at": revision.created_at,
    }
    if include_files:
        payload["files"] = dict(revision.files)
    else:
        payload["file_paths"] = sorted(str(path) for path in revision.files)
    return payload


async def _owned_project(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    event_id: str,
) -> SurfaceProject | None:
    result = await session.execute(
        select(SurfaceProject).where(
            SurfaceProject.organization_id == organization_id,
            SurfaceProject.user_id == user_id,
            SurfaceProject.event_id == event_id,
        )
    )
    return result.scalars().first()


async def _owned_revision(
    session: AsyncSession,
    revision_id: str | None,
    *,
    organization_id: str,
    user_id: str,
    event_id: str,
    project_id: str,
) -> SurfaceRevision | None:
    if revision_id is None:
        return None
    result = await session.execute(
        select(SurfaceRevision).where(
            SurfaceRevision.id == revision_id,
            SurfaceRevision.organization_id == organization_id,
            SurfaceRevision.user_id == user_id,
            SurfaceRevision.event_id == event_id,
            SurfaceRevision.project_id == project_id,
        )
    )
    return result.scalars().first()


async def surface_project_snapshot(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    event_id: str,
    include_files: bool,
) -> dict[str, Any]:
    project = await _owned_project(
        session,
        organization_id=organization_id,
        user_id=user_id,
        event_id=event_id,
    )
    if project is None:
        return {
            "project": None,
            "draft": None,
            "published": None,
            "expected_revision": None,
        }
    draft = await _owned_revision(
        session,
        project.draft_revision_id,
        organization_id=organization_id,
        user_id=user_id,
        event_id=event_id,
        project_id=project.id,
    )
    published = await _owned_revision(
        session,
        project.published_revision_id,
        organization_id=organization_id,
        user_id=user_id,
        event_id=event_id,
        project_id=project.id,
    )
    return {
        "project": {
            "id": project.id,
            "event_id": project.event_id,
            "draft_revision_id": project.draft_revision_id,
            "published_revision_id": project.published_revision_id,
            "sdk_version": project.sdk_version,
            "lifecycle": project.lifecycle,
            "user_lock_state": project.user_lock_state,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
        },
        "draft": _revision_payload(draft, include_files=include_files),
        "published": _revision_payload(published, include_files=include_files),
        "expected_revision": project.draft_revision_id,
    }


class SurfaceAuthoringHandler:
    """Bind Surface authoring to one authenticated Run and Event."""

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

    async def _on_owner_loop(self, coroutine):
        current = asyncio.get_running_loop()
        if self._owner_loop is None or self._owner_loop is current:
            return await coroutine
        future = asyncio.run_coroutine_threadsafe(coroutine, self._owner_loop)
        return await asyncio.wrap_future(future)

    async def _load_event(self, session: AsyncSession) -> Event:
        event_id = self._run_context.event_id
        if not event_id:
            raise SurfaceAuthoringError("Event identity is required")
        event = await session.get(Event, event_id)
        if (
            event is None
            or event.organization_id != self._run_context.organization_id
            or event.user_id != self._run_context.user_id
        ):
            raise SurfaceAuthoringError("Event is unavailable")
        if event.lifecycle == "archived":
            raise SurfaceAuthoringError("Event is archived")
        return event

    async def _read(self) -> dict[str, Any]:
        async with self._session_factory() as session:
            event = await self._load_event(session)
            return await surface_project_snapshot(
                session,
                organization_id=event.organization_id,
                user_id=event.user_id,
                event_id=event.id,
                include_files=True,
            )

    async def read(self) -> dict[str, Any]:
        return await self._on_owner_loop(self._read())

    async def _write(self, params: SurfaceWriteFilesParams) -> dict[str, Any]:
        request_fingerprint = _request_fingerprint(params)
        async with self._session_factory() as session:
            event = await self._load_event(session)
            project = await _owned_project(
                session,
                organization_id=event.organization_id,
                user_id=event.user_id,
                event_id=event.id,
            )
            if project is None:
                if params.expected_revision is not None:
                    raise SurfaceConflictError(
                        "Surface has no draft revision; expected_revision must be null"
                    )
                project = SurfaceProject(
                    organization_id=event.organization_id,
                    user_id=event.user_id,
                    event_id=event.id,
                    sdk_version=SURFACE_SDK_VERSION,
                )
                session.add(project)
                try:
                    await session.flush()
                except IntegrityError as exc:
                    await session.rollback()
                    raise SurfaceConflictError(
                        "Surface project was created concurrently; read it and retry"
                    ) from exc
            if project.user_lock_state != "editable":
                raise SurfaceAuthoringError("Surface source is locked by the user")

            replay_result = await session.execute(
                select(SurfaceRevision).where(
                    SurfaceRevision.project_id == project.id,
                    SurfaceRevision.organization_id == event.organization_id,
                    SurfaceRevision.user_id == event.user_id,
                    SurfaceRevision.idempotency_key == params.idempotency_key,
                )
            )
            replay = replay_result.scalars().first()
            if replay is not None:
                if replay.request_fingerprint != request_fingerprint:
                    raise SurfaceConflictError(
                        "idempotency_key was already used for a different mutation"
                    )
                payload = await surface_project_snapshot(
                    session,
                    organization_id=event.organization_id,
                    user_id=event.user_id,
                    event_id=event.id,
                    include_files=True,
                )
                payload.update({"changed": False, "replayed": True})
                return payload

            current_revision = await _owned_revision(
                session,
                project.draft_revision_id,
                organization_id=event.organization_id,
                user_id=event.user_id,
                event_id=event.id,
                project_id=project.id,
            )
            if project.draft_revision_id != params.expected_revision:
                raise SurfaceConflictError(
                    "Surface draft changed; read the project and retry against "
                    f"{project.draft_revision_id or 'null'}"
                )

            files = (
                {
                    str(path): str(content)
                    for path, content in current_revision.files.items()
                }
                if current_revision is not None
                else {}
            )
            changed_paths: list[str] = []
            for patch in params.patches:
                source_path = _project_source_path(patch.path)
                if patch.operation == "delete":
                    if source_path in files:
                        del files[source_path]
                        changed_paths.append(source_path)
                else:
                    content = patch.content or ""
                    if files.get(source_path) != content:
                        files[source_path] = content
                        changed_paths.append(source_path)

            if len(files) > MAX_SURFACE_FILES:
                raise SurfaceAuthoringError(
                    f"Surface source exceeds {MAX_SURFACE_FILES} files"
                )
            total_bytes = sum(
                len(content.encode("utf-8")) for content in files.values()
            )
            if total_bytes > MAX_SURFACE_SOURCE_BYTES:
                raise SurfaceAuthoringError(
                    f"Surface source exceeds {MAX_SURFACE_SOURCE_BYTES} bytes"
                )
            manifest = {
                "format": "aloy-react-surface",
                "entrypoint": "/src/App.tsx",
                "sdk_version": project.sdk_version,
            }
            revision = SurfaceRevision(
                organization_id=event.organization_id,
                user_id=event.user_id,
                event_id=event.id,
                project_id=project.id,
                revision_number=(
                    current_revision.revision_number + 1
                    if current_revision is not None
                    else 1
                ),
                parent_revision_id=(
                    current_revision.id if current_revision is not None else None
                ),
                creator_run_id=self._run_context.run_id,
                idempotency_key=params.idempotency_key,
                request_fingerprint=request_fingerprint,
                manifest=manifest,
                files=files,
                checksum=_snapshot_checksum(manifest, files),
                file_count=len(files),
                total_bytes=total_bytes,
            )
            session.add(revision)
            try:
                await session.flush()
            except IntegrityError as exc:
                await session.rollback()
                raise SurfaceConflictError(
                    "Surface revision was written concurrently; read it and retry"
                ) from exc

            expected_condition = (
                col(SurfaceProject.draft_revision_id).is_(None)
                if params.expected_revision is None
                else col(SurfaceProject.draft_revision_id) == params.expected_revision
            )
            claimed = await session.execute(
                update(SurfaceProject)
                .where(
                    col(SurfaceProject.id) == project.id,
                    col(SurfaceProject.organization_id) == event.organization_id,
                    col(SurfaceProject.user_id) == event.user_id,
                    col(SurfaceProject.event_id) == event.id,
                    expected_condition,
                )
                .values(
                    draft_revision_id=revision.id,
                    lifecycle="draft",
                    updated_at=datetime.now(timezone.utc),
                )
            )
            if claimed.rowcount != 1:  # type: ignore[attr-defined]
                await session.rollback()
                raise SurfaceConflictError(
                    "Surface draft changed concurrently; read it and retry"
                )

            event.updated_at = datetime.now(timezone.utc)
            session.add(event)
            session.add(
                EventTrailEntry(
                    organization_id=event.organization_id,
                    user_id=event.user_id,
                    event_id=event.id,
                    actor_id=self._run_context.agent_id,
                    kind="surface_draft_changed",
                    summary=f"Updated Surface draft revision {revision.revision_number}",
                    run_id=self._run_context.run_id,
                    evidence_refs=[
                        {"surface_revision_id": revision.id},
                        {"checksum": revision.checksum},
                    ],
                    payload={
                        "project_id": project.id,
                        "revision_id": revision.id,
                        "parent_revision_id": revision.parent_revision_id,
                        "changed_paths": sorted(changed_paths),
                    },
                )
            )
            await session.commit()
            payload = await surface_project_snapshot(
                session,
                organization_id=event.organization_id,
                user_id=event.user_id,
                event_id=event.id,
                include_files=True,
            )
            payload.update({"changed": True, "replayed": False})
            return payload

    async def write(self, params: SurfaceWriteFilesParams) -> dict[str, Any]:
        return await self._on_owner_loop(self._write(params))


__all__ = [
    "MAX_SURFACE_FILES",
    "MAX_SURFACE_FILE_BYTES",
    "MAX_SURFACE_SOURCE_BYTES",
    "SURFACE_SDK_VERSION",
    "SurfaceAuthoringError",
    "SurfaceAuthoringHandler",
    "SurfaceConflictError",
    "SurfaceFilePatch",
    "SurfaceReadProjectParams",
    "SurfaceWriteFilesParams",
    "surface_project_snapshot",
]
