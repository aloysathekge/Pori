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


def replace_virtual_path(
    path: str,
    thread_data: ThreadData,
    prefix: str = VIRTUAL_PREFIX,
) -> str:
    """
    If path starts with prefix, map the first segment (workspace, uploads, outputs)
    to the corresponding path in thread_data; otherwise return path unchanged.
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
    if suffix:
        return os.path.join(base, suffix).replace("/", os.sep)
    return base


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
        if tail:
            return os.path.join(base, tail).replace("/", os.sep)
        return base

    return _VIRTUAL_PATH_PATTERN.sub(replacer, command)
