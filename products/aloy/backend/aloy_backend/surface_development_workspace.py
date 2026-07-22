"""Provider-neutral, Git-tracked workspaces for developing Surface source.

The Event context projection in :mod:`surface_workspace` is canonical input.
This module owns a different boundary: one temporary source checkout where a
Builder may make bounded edits and ask the trusted host to compile them.  No
workspace implementation receives Event database, network, shell, secrets, or
publication authority.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .surface_authoring import (
    MAX_SURFACE_FILE_BYTES,
    MAX_SURFACE_FILES,
    MAX_SURFACE_SOURCE_BYTES,
    surface_source_path,
)
from .surface_build_runner import (
    SurfaceBuildRunner,
    SurfaceBuildRunnerResult,
    validate_surface_source,
)
from .surface_manifest import parse_surface_manifest
from .surface_pipeline import SurfaceCandidate

MAX_WORKSPACE_SEARCH_MATCHES = 100
MAX_WORKSPACE_DIFF_CHARS = 100_000
_WORKSPACE_ID = re.compile(r"[^a-zA-Z0-9_.-]+")


class SurfaceWorkspaceError(ValueError):
    """A workspace request violated the host-owned source contract."""


class SurfaceWorkspaceConflictError(SurfaceWorkspaceError):
    """The workspace changed after the last successful host check."""


class SurfaceWorkspaceEdit(BaseModel):
    """One bounded, provider-neutral source mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["write", "replace_text", "delete"]
    path: str = Field(min_length=1, max_length=500)
    content: str | None = Field(default=None, max_length=MAX_SURFACE_FILE_BYTES)
    match: str | None = Field(default=None, max_length=MAX_SURFACE_FILE_BYTES)
    replacement: str | None = Field(default=None, max_length=MAX_SURFACE_FILE_BYTES)

    @model_validator(mode="after")
    def validate_operation(self) -> "SurfaceWorkspaceEdit":
        if self.operation == "write":
            if (
                self.content is None
                or self.match is not None
                or self.replacement is not None
            ):
                raise ValueError("write requires content and no match or replacement")
        elif self.operation == "delete":
            if (
                self.content is not None
                or self.match is not None
                or self.replacement is not None
            ):
                raise ValueError(
                    "delete does not accept content, match, or replacement"
                )
        elif not self.match or self.replacement is None or self.content is not None:
            raise ValueError(
                "replace_text requires a non-empty match and replacement only"
            )
        _workspace_source_path(self.path)
        return self


class SurfaceWorkspaceCheck(BaseModel):
    """Deterministic feedback from the trusted local toolchain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["succeeded", "failed", "blocked"]
    source_fingerprint: str
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    resource_metrics: dict[str, Any] = Field(default_factory=dict)


class SurfaceWorkspaceReceipt(BaseModel):
    """Immutable evidence for the exact source submitted to publication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: str
    base_commit: str
    candidate_commit: str
    source_fingerprint: str
    changed_paths: list[str]
    diff_excerpt: str


@dataclass(frozen=True)
class FinishedSurfaceWorkspace:
    candidate: SurfaceCandidate
    receipt: SurfaceWorkspaceReceipt


class SurfaceDevelopmentWorkspace(Protocol):
    """Portable Builder contract; remote providers can implement it later."""

    workspace_id: str

    async def list_files(self) -> list[str]: ...

    async def read_file(self, path: str) -> str: ...

    async def search_source(self, query: str) -> list[dict[str, Any]]: ...

    async def apply(self, edits: list[SurfaceWorkspaceEdit]) -> dict[str, Any]: ...

    async def check(self) -> SurfaceWorkspaceCheck: ...

    async def diagnostics(self) -> SurfaceWorkspaceCheck | None: ...

    async def finish(
        self,
        *,
        summary: str,
        primary_jobs: list[str],
    ) -> FinishedSurfaceWorkspace: ...

    async def close(self) -> None: ...


def _source_fingerprint(files: dict[str, str]) -> str:
    encoded = json.dumps(
        files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_workspace_id(value: str) -> str:
    cleaned = _WORKSPACE_ID.sub("-", value).strip(".-")[:80]
    if not cleaned:
        raise SurfaceWorkspaceError("Surface workspace id is invalid")
    return cleaned


def _workspace_source_path(value: str) -> str:
    """Accept public project paths while reusing the canonical authoring jail."""
    candidate = value if value.startswith("/workspace/") else "/workspace" + value
    return surface_source_path(candidate)


class LocalGitSurfaceWorkspace:
    """Local source-only workspace backed by a private temporary Git repo.

    The model never receives its absolute path and cannot execute commands.
    Every Git invocation is a fixed host command with no shell interpolation.
    Compiling is delegated to the configured trusted ``SurfaceBuildRunner``.
    """

    def __init__(
        self,
        *,
        workspace_id: str,
        root: Path,
        build_runner: SurfaceBuildRunner,
    ) -> None:
        self.workspace_id = _safe_workspace_id(workspace_id)
        self._root = root.resolve()
        self._build_runner = build_runner
        self._base_commit = ""
        self._last_check: SurfaceWorkspaceCheck | None = None
        self._closed = False

    @classmethod
    async def create(
        cls,
        *,
        workspace_id: str,
        base_files: dict[str, str],
        build_runner: SurfaceBuildRunner,
        parent: Path | None = None,
    ) -> "LocalGitSurfaceWorkspace":
        safe_id = _safe_workspace_id(workspace_id)
        parent_root = parent or Path(tempfile.gettempdir()) / "aloy-surface-workspaces"
        parent_root.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix=f"{safe_id}-", dir=parent_root)).resolve()
        workspace = cls(
            workspace_id=safe_id,
            root=root,
            build_runner=build_runner,
        )
        try:
            await asyncio.to_thread(workspace._initialize, base_files)
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise
        return workspace

    def _require_open(self) -> None:
        if self._closed:
            raise SurfaceWorkspaceError("Surface workspace is closed")

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            ["git", *args],
            cwd=self._root,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env={
                **os.environ,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_TERMINAL_PROMPT": "0",
            },
        )
        if check and completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[:2_000]
            raise SurfaceWorkspaceError(
                f"Surface workspace Git operation failed: {detail}"
            )
        return completed

    def _target(self, path: str) -> tuple[str, Path]:
        normalized = _workspace_source_path(path)
        target = (self._root / normalized.lstrip("/")).resolve()
        if self._root not in target.parents:
            raise SurfaceWorkspaceError("Surface source path escapes its workspace")
        if target == self._root / ".git" or self._root / ".git" in target.parents:
            raise SurfaceWorkspaceError(
                "Surface source cannot access workspace metadata"
            )
        return normalized, target

    @staticmethod
    def _validate_limits(files: dict[str, str]) -> None:
        if len(files) > MAX_SURFACE_FILES:
            raise SurfaceWorkspaceError("Surface source contains too many files")
        total = 0
        for path, content in files.items():
            encoded = content.encode("utf-8")
            if len(encoded) > MAX_SURFACE_FILE_BYTES:
                raise SurfaceWorkspaceError(f"Surface source file is too large: {path}")
            total += len(encoded)
        if total > MAX_SURFACE_SOURCE_BYTES:
            raise SurfaceWorkspaceError("Surface source exceeds its total size limit")

    def _write_snapshot(self, files: dict[str, str]) -> None:
        self._validate_limits(files)
        existing = {
            "/" + path.relative_to(self._root).as_posix()
            for path in self._root.rglob("*")
            if path.is_file() and self._root / ".git" not in path.parents
        }
        for path in existing - set(files):
            _, target = self._target(path)
            target.unlink(missing_ok=True)
        for path, content in files.items():
            normalized, target = self._target(path)
            if normalized != path:
                raise SurfaceWorkspaceError(
                    f"Surface source path is not canonical: {path}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".aloy-tmp")
            temporary.write_text(content, encoding="utf-8", newline="\n")
            os.replace(temporary, target)

    def _files(self) -> dict[str, str]:
        self._require_open()
        files: dict[str, str] = {}
        for path in sorted(self._root.rglob("*")):
            if not path.is_file() or self._root / ".git" in path.parents:
                continue
            files["/" + path.relative_to(self._root).as_posix()] = path.read_text(
                encoding="utf-8"
            )
        self._validate_limits(files)
        return files

    def _initialize(self, base_files: dict[str, str]) -> None:
        normalized = {
            _workspace_source_path(path): str(content)
            for path, content in base_files.items()
        }
        self._write_snapshot(normalized)
        self._git("init", "--quiet")
        self._git("config", "user.name", "Aloy Surface Builder")
        self._git("config", "user.email", "surface-builder@aloy.local")
        self._git("config", "commit.gpgsign", "false")
        self._git("add", "--all")
        self._git("commit", "--quiet", "--allow-empty", "-m", "surface: baseline")
        self._base_commit = self._git("rev-parse", "HEAD").stdout.strip()

    async def list_files(self) -> list[str]:
        return sorted((await asyncio.to_thread(self._files)).keys())

    async def read_file(self, path: str) -> str:
        def read() -> str:
            self._require_open()
            _, target = self._target(path)
            if not target.is_file():
                raise SurfaceWorkspaceError(
                    f"Surface source file does not exist: {path}"
                )
            return target.read_text(encoding="utf-8")

        return await asyncio.to_thread(read)

    async def search_source(self, query: str) -> list[dict[str, Any]]:
        cleaned = query.strip()
        if not cleaned or len(cleaned) > 500:
            raise SurfaceWorkspaceError("Surface source query is invalid")

        def search() -> list[dict[str, Any]]:
            matches: list[dict[str, Any]] = []
            for path, content in self._files().items():
                for line_number, line in enumerate(content.splitlines(), start=1):
                    if cleaned.casefold() not in line.casefold():
                        continue
                    matches.append(
                        {
                            "path": path,
                            "line": line_number,
                            "excerpt": line[:500],
                        }
                    )
                    if len(matches) >= MAX_WORKSPACE_SEARCH_MATCHES:
                        return matches
            return matches

        return await asyncio.to_thread(search)

    async def apply(self, edits: list[SurfaceWorkspaceEdit]) -> dict[str, Any]:
        if not edits:
            raise SurfaceWorkspaceError("Surface workspace edit batch is empty")

        def mutate() -> dict[str, Any]:
            before = self._files()
            after = dict(before)
            touched: list[str] = []
            for edit in edits:
                path = _workspace_source_path(edit.path)
                if edit.operation == "delete":
                    if path not in after:
                        raise SurfaceWorkspaceError(
                            f"Cannot delete missing Surface source file: {path}"
                        )
                    del after[path]
                elif edit.operation == "replace_text":
                    if path not in after:
                        raise SurfaceWorkspaceError(
                            f"Cannot replace text in missing Surface source file: {path}"
                        )
                    assert edit.match is not None
                    assert edit.replacement is not None
                    occurrences = after[path].count(edit.match)
                    if occurrences != 1:
                        raise SurfaceWorkspaceError(
                            "replace_text match must occur exactly once in "
                            f"{path}; found {occurrences} occurrences"
                        )
                    after[path] = after[path].replace(edit.match, edit.replacement, 1)
                else:
                    assert edit.content is not None
                    after[path] = edit.content
                if path not in touched:
                    touched.append(path)
            if after == before:
                raise SurfaceWorkspaceError(
                    "Surface workspace edit does not change source"
                )
            self._write_snapshot(after)
            self._last_check = None
            return {
                "changed_paths": touched,
                "source_fingerprint": _source_fingerprint(after),
            }

        return await asyncio.to_thread(mutate)

    async def check(self) -> SurfaceWorkspaceCheck:
        files = await asyncio.to_thread(self._files)
        fingerprint = _source_fingerprint(files)
        diagnostics: list[dict[str, Any]] = []
        manifest: dict[str, Any] = {}
        try:
            manifest = json.loads(files.get("/surface.json", "{}"))
            if not isinstance(manifest, dict):
                raise ValueError("surface.json must contain an object")
            parse_surface_manifest(files)
        except (TypeError, ValueError) as exc:
            diagnostics.append(
                {
                    "stage": "validation",
                    "code": "invalid_manifest",
                    "severity": "error",
                    "message": str(exc),
                    "path": "/surface.json",
                }
            )
        diagnostics.extend(validate_surface_source(files, manifest))
        if diagnostics:
            result = SurfaceWorkspaceCheck(
                status="failed",
                source_fingerprint=fingerprint,
                diagnostics=diagnostics[:100],
            )
        else:
            build: SurfaceBuildRunnerResult = await self._build_runner.build(
                build_id=f"workspace-{self.workspace_id}-{fingerprint[:12]}",
                files=files,
                manifest=manifest,
            )
            result = SurfaceWorkspaceCheck(
                status=build.status,
                source_fingerprint=fingerprint,
                diagnostics=list(build.diagnostics)[:100],
                resource_metrics=dict(build.resource_metrics),
            )
        self._last_check = result
        return result

    async def diagnostics(self) -> SurfaceWorkspaceCheck | None:
        return self._last_check

    async def finish(
        self,
        *,
        summary: str,
        primary_jobs: list[str],
    ) -> FinishedSurfaceWorkspace:
        files = await asyncio.to_thread(self._files)
        fingerprint = _source_fingerprint(files)
        if (
            self._last_check is None
            or self._last_check.status != "succeeded"
            or self._last_check.source_fingerprint != fingerprint
        ):
            raise SurfaceWorkspaceConflictError(
                "Surface source must pass the trusted check after its last edit"
            )

        def commit() -> tuple[str, list[str], str]:
            self._git("add", "--all")
            self._git(
                "commit",
                "--quiet",
                "--allow-empty",
                "-m",
                "surface: candidate",
            )
            candidate_commit = self._git("rev-parse", "HEAD").stdout.strip()
            changed = [
                line
                for line in self._git(
                    "diff",
                    "--name-only",
                    self._base_commit,
                    candidate_commit,
                ).stdout.splitlines()
                if line
            ]
            diff = self._git(
                "diff",
                "--no-ext-diff",
                "--unified=3",
                self._base_commit,
                candidate_commit,
            ).stdout[:MAX_WORKSPACE_DIFF_CHARS]
            return candidate_commit, sorted(changed), diff

        candidate_commit, changed_paths, diff = await asyncio.to_thread(commit)
        candidate = SurfaceCandidate.model_validate(
            {
                "summary": summary,
                "primary_jobs": primary_jobs,
                "files": [
                    {"path": f"/workspace{path}", "content": content}
                    for path, content in sorted(files.items())
                ],
            }
        )
        return FinishedSurfaceWorkspace(
            candidate=candidate,
            receipt=SurfaceWorkspaceReceipt(
                workspace_id=self.workspace_id,
                base_commit=self._base_commit,
                candidate_commit=candidate_commit,
                source_fingerprint=fingerprint,
                changed_paths=changed_paths,
                diff_excerpt=diff,
            ),
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(shutil.rmtree, self._root, True)


__all__ = [
    "FinishedSurfaceWorkspace",
    "LocalGitSurfaceWorkspace",
    "SurfaceDevelopmentWorkspace",
    "SurfaceWorkspaceCheck",
    "SurfaceWorkspaceConflictError",
    "SurfaceWorkspaceEdit",
    "SurfaceWorkspaceError",
    "SurfaceWorkspaceReceipt",
]
