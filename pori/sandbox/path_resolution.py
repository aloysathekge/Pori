"""
Virtual path resolution and thread data for local sandbox.

When the sandbox runs on the host (no real container), tools must rewrite
virtual paths (e.g. /mnt/user-data/workspace) to real paths before calling
the sandbox. Use these helpers only for LocalSandboxProvider.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Default virtual prefix; agent and system prompt should use these paths
VIRTUAL_PREFIX = "/mnt/user-data"
WORKSPACE_SEGMENT = "workspace"
UPLOADS_SEGMENT = "uploads"
OUTPUTS_SEGMENT = "outputs"


@dataclass
class ThreadData:
    """Per-thread paths on the host for workspace, uploads, and outputs."""

    workspace_path: str
    uploads_path: str
    outputs_path: str

    def ensure_dirs(self) -> None:
        """Create the three directories if they do not exist."""
        for p in (self.workspace_path, self.uploads_path, self.outputs_path):
            Path(p).mkdir(parents=True, exist_ok=True)


def get_thread_data(
    thread_id: str,
    base_dir: str,
    virtual_prefix: str = VIRTUAL_PREFIX,
) -> ThreadData:
    """
    Compute thread data for a given thread_id and ensure directories exist.
    Layout: {base_dir}/threads/{thread_id}/user-data/{workspace,uploads,outputs}
    """
    root = Path(base_dir) / "threads" / thread_id / "user-data"
    data = ThreadData(
        workspace_path=str(root / WORKSPACE_SEGMENT),
        uploads_path=str(root / UPLOADS_SEGMENT),
        outputs_path=str(root / OUTPUTS_SEGMENT),
    )
    data.ensure_dirs()
    return data


def get_workspace_data(
    workspace_id: str,
    run_id: str,
    base_dir: str,
) -> ThreadData:
    """Return an Event workspace with isolated scratch for one run."""
    event_root = Path(base_dir) / "events" / workspace_id
    user_data = event_root / "user-data"
    data = ThreadData(
        workspace_path=str(user_data / WORKSPACE_SEGMENT),
        uploads_path=str(user_data / UPLOADS_SEGMENT),
        outputs_path=str(event_root / "runs" / run_id / "scratch"),
    )
    data.ensure_dirs()
    return data


def _resolve_existing(p: Path) -> Path:
    """Resolve ``p`` even when it doesn't exist yet.

    Resolves the nearest existing ancestor (following symlinks — the part
    ``normpath`` can't do) and re-appends the not-yet-created leaf segments, so a
    validation works for both existing and about-to-be-created targets and can't
    be fooled by a symlink in the existing prefix.
    """
    missing: list[str] = []
    cur = p
    while not cur.exists():
        parent = cur.parent
        if parent == cur:  # reached the filesystem root
            break
        missing.append(cur.name)
        cur = parent
    resolved = cur.resolve()
    for name in reversed(missing):
        resolved = resolved / name
    return resolved


def _safe_join(base: str, suffix: str) -> str:
    """
    Join suffix onto base and guarantee the result stays within base.

    Two layers: a fast string pre-check (``normpath`` + prefix) that cheaply
    rejects obvious ``../`` escapes in the suffix, then a symlink-safe check that
    ``resolve()``s both sides and asserts containment with ``relative_to`` — so a
    symlink *inside* the sandbox pointing outside it (which ``normpath`` follows
    blindly) is also blocked. Raises ValueError on any escape.
    """
    base_norm = os.path.normpath(base)
    if not suffix:
        return base_norm
    candidate = os.path.normpath(os.path.join(base_norm, suffix))
    if candidate != base_norm and not candidate.startswith(base_norm + os.sep):
        raise ValueError(
            f"Path traversal blocked: {suffix!r} escapes the sandbox directory"
        )
    # Symlink-following containment check (normpath does not chase links).
    base_resolved = Path(base_norm).resolve()
    cand_resolved = _resolve_existing(Path(candidate))
    try:
        cand_resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError(
            f"Path traversal blocked: {suffix!r} escapes the sandbox directory "
            "(symlink)"
        )
    return candidate


def replace_virtual_path(
    path: str,
    thread_data: ThreadData,
    prefix: str = VIRTUAL_PREFIX,
) -> str:
    """
    If path starts with prefix, map the first segment (workspace, uploads, outputs)
    to the corresponding path in thread_data; otherwise return path unchanged.

    Raises ValueError if the path attempts to traverse outside the sandbox dir.
    """
    path = path.strip()
    if not path.startswith(prefix):
        return path
    rest = path[len(prefix) :].lstrip("/")
    segment = rest.split("/")[0] if rest else ""
    if segment == WORKSPACE_SEGMENT:
        base = thread_data.workspace_path
    elif segment == UPLOADS_SEGMENT:
        base = thread_data.uploads_path
    elif segment == OUTPUTS_SEGMENT:
        base = thread_data.outputs_path
    else:
        return path
    suffix = rest[len(segment) :].lstrip("/") if len(rest) > len(segment) else ""
    return _safe_join(base, suffix)


def to_virtual_path(
    real_path: str,
    thread_data: ThreadData,
    prefix: str = VIRTUAL_PREFIX,
) -> str:
    """
    Map a real thread path back to its virtual form (e.g. .../outputs/a.zip ->
    /mnt/user-data/outputs/a.zip) for display. Inverse of replace_virtual_path.

    Returns real_path unchanged if it is not under a known thread directory.
    """
    norm = os.path.normpath(real_path)
    for segment, base in (
        (WORKSPACE_SEGMENT, thread_data.workspace_path),
        (UPLOADS_SEGMENT, thread_data.uploads_path),
        (OUTPUTS_SEGMENT, thread_data.outputs_path),
    ):
        base_norm = os.path.normpath(base)
        if norm == base_norm:
            return f"{prefix}/{segment}"
        if norm.startswith(base_norm + os.sep):
            rest = norm[len(base_norm) + 1 :].replace(os.sep, "/")
            return f"{prefix}/{segment}/{rest}"
    return real_path


# Pattern: prefix followed by optional path (e.g. /workspace, /workspace/foo/bar)
_VIRTUAL_PATH_PATTERN = re.compile(
    re.escape(VIRTUAL_PREFIX) + r"(/[^\s'\"]*)?",
    re.IGNORECASE,
)


def replace_virtual_paths_in_command(
    command: str,
    thread_data: ThreadData,
    prefix: str = VIRTUAL_PREFIX,
) -> str:
    """
    Find all occurrences of the virtual prefix followed by a path in the command
    string and replace each with the resolved real path. Use for local sandbox
    so bash commands with /mnt/user-data/... hit the right dirs.
    """

    def replacer(match: re.Match) -> str:
        full = match.group(0)
        if full == prefix or full.rstrip("/") == prefix:
            return thread_data.workspace_path
        suffix = full[len(prefix) :].lstrip("/")
        segment = suffix.split("/")[0] if suffix else ""
        if segment == WORKSPACE_SEGMENT:
            base = thread_data.workspace_path
        elif segment == UPLOADS_SEGMENT:
            base = thread_data.uploads_path
        elif segment == OUTPUTS_SEGMENT:
            base = thread_data.outputs_path
        else:
            return full
        tail = suffix[len(segment) :].lstrip("/") if len(suffix) > len(segment) else ""
        try:
            return _safe_join(base, tail)
        except ValueError:
            # Leave a traversal attempt untouched rather than rewriting it to an
            # escaped real path; the sandbox shell will then fail safely on it.
            return full

    return _VIRTUAL_PATH_PATTERN.sub(replacer, command)
