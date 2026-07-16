"""Typed, deny-by-default virtual file backends for product-scoped runs.

The contract deliberately excludes command execution. Shell access remains a
separate ``Sandbox`` capability so a virtual mount can never expand process or
host access merely because it is readable by file tools.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from .sandbox.base import Sandbox

FILE_BACKEND_CONTEXT_KEY = "file_backend"


class _UnmountedPathError(ValueError):
    """Raised when a valid virtual path has no authorized mount."""


class FileErrorCode(str, Enum):
    NOT_FOUND = "file_not_found"
    PERMISSION_DENIED = "permission_denied"
    INVALID_PATH = "invalid_path"
    IS_DIRECTORY = "is_directory"
    NOT_DIRECTORY = "not_directory"
    CONFLICT = "conflict"
    UNSUPPORTED = "unsupported"
    BACKEND_ERROR = "backend_error"


class FileEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    is_dir: bool = False
    size: Optional[int] = Field(default=None, ge=0)


class FileListResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    entries: Tuple[FileEntry, ...] = ()
    error_code: Optional[FileErrorCode] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error_code is None


class FileReadResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    content: Optional[str] = None
    encoding: str = "utf-8"
    total_lines: Optional[int] = Field(default=None, ge=0)
    start_line: Optional[int] = Field(default=None, ge=1)
    end_line: Optional[int] = Field(default=None, ge=1)
    next_offset: Optional[int] = Field(default=None, ge=0)
    error_code: Optional[FileErrorCode] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error_code is None


class FileMutationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    changed: bool = False
    bytes_written: int = Field(default=0, ge=0)
    replacements: int = Field(default=0, ge=0)
    error_code: Optional[FileErrorCode] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error_code is None


def normalize_virtual_path(path: str) -> str:
    """Return one canonical POSIX path and reject traversal or host syntax."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Virtual file paths must be non-empty strings")
    value = path.strip()
    if "\x00" in value or "\\" in value:
        raise ValueError("Virtual file paths cannot contain NUL or backslashes")
    if not value.startswith("/"):
        raise ValueError("Virtual file paths must start with '/'")
    raw_parts = value.split("/")
    if any(part in {"..", "~"} for part in raw_parts):
        raise ValueError("Virtual file path traversal is not allowed")
    parts = [part for part in raw_parts if part not in {"", "."}]
    return "/" + "/".join(parts) if parts else "/"


def _read_window(
    path: str, content: str, offset: int, limit: Optional[int]
) -> FileReadResult:
    if offset < 0:
        return FileReadResult(
            path=path,
            error_code=FileErrorCode.INVALID_PATH,
            error="offset must be non-negative",
        )
    if limit is not None and limit <= 0:
        return FileReadResult(
            path=path,
            error_code=FileErrorCode.INVALID_PATH,
            error="limit must be positive when provided",
        )
    lines = content.splitlines()
    total = len(lines)
    if offset == 0 and limit is None:
        return FileReadResult(
            path=path,
            content=content,
            total_lines=total,
            start_line=1 if total else None,
            end_line=total if total else None,
        )
    window = lines[offset:] if limit is None else lines[offset : offset + limit]
    if not window:
        return FileReadResult(path=path, content="", total_lines=total)
    end_offset = offset + len(window)
    return FileReadResult(
        path=path,
        content="\n".join(window),
        total_lines=total,
        start_line=offset + 1,
        end_line=end_offset,
        next_offset=end_offset if end_offset < total else None,
    )


class FileBackend(ABC):
    """Storage-only virtual filesystem contract; never a shell capability."""

    @abstractmethod
    def list(
        self,
        path: str,
        *,
        recursive: bool = False,
        max_depth: int = 3,
    ) -> FileListResult: ...

    @abstractmethod
    def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> FileReadResult: ...

    @abstractmethod
    def write(
        self,
        path: str,
        content: str,
        *,
        append: bool = False,
    ) -> FileMutationResult: ...

    def edit(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
    ) -> FileMutationResult:
        if not old:
            return FileMutationResult(
                path=path,
                error_code=FileErrorCode.CONFLICT,
                error="old text must not be empty",
            )
        result = self.read(path)
        if not result.success or result.content is None:
            return FileMutationResult(
                path=path,
                error_code=result.error_code or FileErrorCode.BACKEND_ERROR,
                error=result.error or "file could not be read",
            )
        occurrences = result.content.count(old)
        if occurrences == 0:
            return FileMutationResult(
                path=path,
                error_code=FileErrorCode.CONFLICT,
                error="old text was not found",
            )
        if occurrences > 1 and not replace_all:
            return FileMutationResult(
                path=path,
                error_code=FileErrorCode.CONFLICT,
                error=f"old text appears {occurrences} times",
            )
        updated = (
            result.content.replace(old, new)
            if replace_all
            else result.content.replace(old, new, 1)
        )
        written = self.write(path, updated)
        if not written.success:
            return written
        return written.model_copy(
            update={"replacements": occurrences if replace_all else 1}
        )


class MemoryFileBackend(FileBackend):
    """Thread-safe ephemeral backend useful for projections and run scratch."""

    def __init__(self, files: Optional[Mapping[str, str]] = None):
        self._files = {
            normalize_virtual_path(path): content
            for path, content in dict(files or {}).items()
        }
        self._lock = threading.RLock()

    def list(
        self,
        path: str,
        *,
        recursive: bool = False,
        max_depth: int = 3,
    ) -> FileListResult:
        try:
            directory = normalize_virtual_path(path)
        except ValueError as exc:
            return FileListResult(
                path=str(path),
                error_code=FileErrorCode.INVALID_PATH,
                error=str(exc),
            )
        if max_depth < 0:
            return FileListResult(
                path=directory,
                error_code=FileErrorCode.INVALID_PATH,
                error="max_depth must be non-negative",
            )
        prefix = "/" if directory == "/" else directory + "/"
        entries: dict[str, FileEntry] = {}
        with self._lock:
            if directory in self._files:
                return FileListResult(
                    path=directory,
                    error_code=FileErrorCode.NOT_DIRECTORY,
                    error=f"Path is not a directory: {directory}",
                )
            for file_path, content in self._files.items():
                if not file_path.startswith(prefix):
                    continue
                relative = file_path[len(prefix) :]
                if not relative:
                    continue
                parts = relative.split("/")
                depth = len(parts) - 1
                if recursive:
                    for index in range(min(len(parts) - 1, max_depth)):
                        dir_path = (
                            prefix.rstrip("/") + "/" + "/".join(parts[: index + 1])
                        )
                        entries[dir_path] = FileEntry(path=dir_path, is_dir=True)
                    if depth <= max_depth:
                        entries[file_path] = FileEntry(
                            path=file_path,
                            size=len(content.encode("utf-8")),
                        )
                elif len(parts) == 1:
                    entries[file_path] = FileEntry(
                        path=file_path,
                        size=len(content.encode("utf-8")),
                    )
                else:
                    dir_path = prefix.rstrip("/") + "/" + parts[0]
                    entries[dir_path] = FileEntry(path=dir_path, is_dir=True)
        return FileListResult(
            path=directory,
            entries=tuple(entries[key] for key in sorted(entries)),
        )

    def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> FileReadResult:
        try:
            normalized = normalize_virtual_path(path)
        except ValueError as exc:
            return FileReadResult(
                path=str(path),
                error_code=FileErrorCode.INVALID_PATH,
                error=str(exc),
            )
        with self._lock:
            content = self._files.get(normalized)
        if content is None:
            return FileReadResult(
                path=normalized,
                error_code=FileErrorCode.NOT_FOUND,
                error=f"File does not exist: {normalized}",
            )
        return _read_window(normalized, content, offset, limit)

    def write(
        self,
        path: str,
        content: str,
        *,
        append: bool = False,
    ) -> FileMutationResult:
        try:
            normalized = normalize_virtual_path(path)
        except ValueError as exc:
            return FileMutationResult(
                path=str(path),
                error_code=FileErrorCode.INVALID_PATH,
                error=str(exc),
            )
        if normalized == "/":
            return FileMutationResult(
                path=normalized,
                error_code=FileErrorCode.IS_DIRECTORY,
                error="Cannot write to the virtual root",
            )
        with self._lock:
            previous = self._files.get(normalized, "") if append else ""
            self._files[normalized] = previous + content
        return FileMutationResult(
            path=normalized,
            changed=True,
            bytes_written=len(content.encode("utf-8")),
        )


class SandboxFileBackend(FileBackend):
    """Adapt an existing isolated Pori ``Sandbox`` as file-only storage."""

    def __init__(self, sandbox: Sandbox, *, root: str = "/"):
        self.sandbox = sandbox
        self.root = normalize_virtual_path(root)

    def _sandbox_path(self, path: str) -> str:
        normalized = normalize_virtual_path(path)
        suffix = normalized.lstrip("/")
        return self.root.rstrip("/") + ("/" + suffix if suffix else "") or "/"

    def list(
        self,
        path: str,
        *,
        recursive: bool = False,
        max_depth: int = 3,
    ) -> FileListResult:
        try:
            normalized = normalize_virtual_path(path)
        except ValueError as exc:
            return FileListResult(
                path=str(path),
                error_code=FileErrorCode.INVALID_PATH,
                error=str(exc),
            )
        if max_depth < 0:
            return FileListResult(
                path=normalized,
                error_code=FileErrorCode.INVALID_PATH,
                error="max_depth must be non-negative",
            )
        try:
            resolved = self._sandbox_path(normalized)
            depth = max_depth if recursive else 1
            raw_entries = self.sandbox.list_dir(resolved, max_depth=depth)
            entries = []
            for raw in raw_entries:
                value = raw.replace("\\", "/").strip()
                is_dir = value.endswith("/")
                value = value.rstrip("/")
                if not value:
                    continue
                entry_path = normalize_virtual_path(
                    normalized.rstrip("/") + "/" + value.lstrip("/")
                )
                entries.append(FileEntry(path=entry_path, is_dir=is_dir))
            return FileListResult(
                path=normalized,
                entries=tuple(sorted(entries, key=lambda item: item.path)),
            )
        except Exception as exc:
            return FileListResult(
                path=str(path),
                error_code=FileErrorCode.BACKEND_ERROR,
                error=str(exc),
            )

    def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> FileReadResult:
        try:
            normalized = normalize_virtual_path(path)
        except ValueError as exc:
            return FileReadResult(
                path=str(path),
                error_code=FileErrorCode.INVALID_PATH,
                error=str(exc),
            )
        try:
            content = self.sandbox.read_file(self._sandbox_path(normalized))
        except Exception as exc:
            return FileReadResult(
                path=str(path),
                error_code=FileErrorCode.BACKEND_ERROR,
                error=str(exc),
            )
        return _read_window(normalized, content, offset, limit)

    def write(
        self,
        path: str,
        content: str,
        *,
        append: bool = False,
    ) -> FileMutationResult:
        try:
            normalized = normalize_virtual_path(path)
        except ValueError as exc:
            return FileMutationResult(
                path=str(path),
                error_code=FileErrorCode.INVALID_PATH,
                error=str(exc),
            )
        try:
            self.sandbox.write_file(
                self._sandbox_path(normalized),
                content,
                append=append,
            )
            return FileMutationResult(
                path=normalized,
                changed=True,
                bytes_written=len(content.encode("utf-8")),
            )
        except Exception as exc:
            return FileMutationResult(
                path=str(path),
                error_code=FileErrorCode.BACKEND_ERROR,
                error=str(exc),
            )


class ReadOnlyFileBackend(FileBackend):
    """Expose another backend without granting mutations."""

    def __init__(self, backend: FileBackend):
        self.backend = backend

    def list(
        self,
        path: str,
        *,
        recursive: bool = False,
        max_depth: int = 3,
    ) -> FileListResult:
        return self.backend.list(path, recursive=recursive, max_depth=max_depth)

    def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> FileReadResult:
        return self.backend.read(path, offset=offset, limit=limit)

    def write(
        self,
        path: str,
        content: str,
        *,
        append: bool = False,
    ) -> FileMutationResult:
        return FileMutationResult(
            path=path,
            error_code=FileErrorCode.PERMISSION_DENIED,
            error=f"Read-only file backend denies writes to {path}",
        )

    def edit(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
    ) -> FileMutationResult:
        return FileMutationResult(
            path=path,
            error_code=FileErrorCode.PERMISSION_DENIED,
            error=f"Read-only file backend denies edits to {path}",
        )


@dataclass(frozen=True)
class FileMount:
    prefix: str
    backend: FileBackend
    read_only: bool = False

    def __post_init__(self) -> None:
        normalized = normalize_virtual_path(self.prefix)
        if normalized == "/":
            raise ValueError("Use CompositeFileBackend.default for the root mount")
        object.__setattr__(self, "prefix", normalized)


class CompositeFileBackend(FileBackend):
    """Route virtual paths by longest mount prefix; unmatched paths are denied."""

    def __init__(
        self,
        mounts: Tuple[FileMount, ...],
        *,
        default: Optional[FileBackend] = None,
    ):
        prefixes = [mount.prefix for mount in mounts]
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("Composite file mount prefixes must be unique")
        self.mounts = tuple(
            sorted(mounts, key=lambda mount: len(mount.prefix), reverse=True)
        )
        self.default = default

    def _route(self, path: str) -> tuple[FileBackend, str, Optional[FileMount]]:
        normalized = normalize_virtual_path(path)
        for mount in self.mounts:
            if normalized == mount.prefix:
                return mount.backend, "/", mount
            if normalized.startswith(mount.prefix + "/"):
                internal = normalized[len(mount.prefix) :]
                return mount.backend, internal, mount
        if self.default is not None:
            return self.default, normalized, None
        raise _UnmountedPathError(f"No virtual file mount handles {normalized}")

    @staticmethod
    def _external_path(path: str, mount: Optional[FileMount]) -> str:
        if mount is None:
            return path
        normalized = normalize_virtual_path(path)
        return mount.prefix if normalized == "/" else mount.prefix + normalized

    def _virtual_children(self, directory: str) -> Tuple[FileEntry, ...]:
        """Return immediate mount directories visible below ``directory``."""
        normalized = normalize_virtual_path(directory)
        prefix = "/" if normalized == "/" else normalized + "/"
        children: dict[str, FileEntry] = {}
        for mount in self.mounts:
            if not mount.prefix.startswith(prefix):
                continue
            remainder = mount.prefix[len(prefix) :]
            if not remainder:
                continue
            child_name = remainder.split("/", 1)[0]
            child_path = prefix.rstrip("/") + "/" + child_name
            children[child_path] = FileEntry(path=child_path, is_dir=True)
        return tuple(children[path] for path in sorted(children))

    def _route_error(self, path: str, exc: ValueError, result_type):
        return result_type(
            path=str(path),
            error_code=(
                FileErrorCode.PERMISSION_DENIED
                if isinstance(exc, _UnmountedPathError)
                else FileErrorCode.INVALID_PATH
            ),
            error=str(exc),
        )

    def list(
        self,
        path: str,
        *,
        recursive: bool = False,
        max_depth: int = 3,
    ) -> FileListResult:
        try:
            normalized = normalize_virtual_path(path)
        except ValueError as exc:
            return self._route_error(path, exc, FileListResult)
        virtual_children = self._virtual_children(normalized)
        try:
            backend, internal, mount = self._route(path)
        except ValueError as exc:
            if virtual_children:
                return FileListResult(path=normalized, entries=virtual_children)
            return self._route_error(path, exc, FileListResult)
        result = backend.list(internal, recursive=recursive, max_depth=max_depth)
        if not result.success:
            return result.model_copy(update={"path": normalize_virtual_path(path)})
        entries = {
            self._external_path(entry.path, mount): entry.model_copy(
                update={"path": self._external_path(entry.path, mount)}
            )
            for entry in result.entries
        }
        entries.update({entry.path: entry for entry in virtual_children})
        return result.model_copy(
            update={
                "path": normalized,
                "entries": tuple(entries[path] for path in sorted(entries)),
            }
        )

    def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> FileReadResult:
        try:
            backend, internal, mount = self._route(path)
        except ValueError as exc:
            return self._route_error(path, exc, FileReadResult)
        result = backend.read(internal, offset=offset, limit=limit)
        return result.model_copy(
            update={"path": self._external_path(result.path, mount)}
        )

    def write(
        self,
        path: str,
        content: str,
        *,
        append: bool = False,
    ) -> FileMutationResult:
        try:
            backend, internal, mount = self._route(path)
        except ValueError as exc:
            return self._route_error(path, exc, FileMutationResult)
        if mount is not None and mount.read_only:
            return FileMutationResult(
                path=normalize_virtual_path(path),
                error_code=FileErrorCode.PERMISSION_DENIED,
                error=f"Mount {mount.prefix} is read-only",
            )
        result = backend.write(internal, content, append=append)
        return result.model_copy(
            update={"path": self._external_path(result.path, mount)}
        )

    def edit(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
    ) -> FileMutationResult:
        try:
            backend, internal, mount = self._route(path)
        except ValueError as exc:
            return self._route_error(path, exc, FileMutationResult)
        if mount is not None and mount.read_only:
            return FileMutationResult(
                path=normalize_virtual_path(path),
                error_code=FileErrorCode.PERMISSION_DENIED,
                error=f"Mount {mount.prefix} is read-only",
            )
        result = backend.edit(
            internal,
            old,
            new,
            replace_all=replace_all,
        )
        return result.model_copy(
            update={"path": self._external_path(result.path, mount)}
        )


__all__ = [
    "FILE_BACKEND_CONTEXT_KEY",
    "CompositeFileBackend",
    "FileBackend",
    "FileEntry",
    "FileErrorCode",
    "FileListResult",
    "FileMount",
    "FileMutationResult",
    "FileReadResult",
    "MemoryFileBackend",
    "ReadOnlyFileBackend",
    "SandboxFileBackend",
    "normalize_virtual_path",
]
