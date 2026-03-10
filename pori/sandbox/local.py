"""
Local (host) sandbox implementation.

Runs commands and file I/O on the host. Path mappings for shared dirs
(e.g. /mnt/skills) only; thread-specific paths (/mnt/user-data/...) are
resolved in the tool layer and passed in already resolved.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

from .base import Sandbox, SandboxProvider


class LocalSandbox(Sandbox):
    """
    Sandbox that runs on the host. execute_command uses subprocess;
    file ops use Python I/O. Optional path_mappings for shared dirs
    (e.g. /mnt/skills -> host path). Thread paths are resolved by tools.
    """

    def __init__(self, path_mappings: Optional[Dict[str, str]] = None):
        self.path_mappings = path_mappings or {}

    def _resolve_path(self, path: str) -> str:
        """Apply shared path mappings only (thread paths come in already resolved)."""
        path = path.strip()
        for virtual, real in self.path_mappings.items():
            if (
                path == virtual
                or path.startswith(virtual + "/")
                or path.startswith(virtual + "\\")
            ):
                return os.path.join(real, path[len(virtual) :].lstrip("/\\")).replace(
                    "/", os.sep
                )
        return path

    def execute_command(self, command: str) -> str:
        """Run command with subprocess; command may already have paths resolved by tools."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            out = result.stdout or ""
            err = result.stderr or ""
            if result.returncode != 0:
                return f"stdout:\n{out}\nstderr:\n{err}\n exit_code={result.returncode}"
            return out + ("\n" + err if err else "")
        except subprocess.TimeoutExpired:
            return "Error: command timed out (300s)"
        except Exception as e:
            return f"Error: {e}"

    def read_file(self, path: str) -> str:
        """Read file as text. Path may be already resolved or use path_mappings."""
        resolved = self._resolve_path(path)
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def write_file(self, path: str, content: str, append: bool = False) -> None:
        """Write or append text to file."""
        resolved = self._resolve_path(path)
        Path(resolved).parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(resolved, mode, encoding="utf-8") as f:
            f.write(content)

    def list_dir(self, path: str, max_depth: int = 2) -> list[str]:
        """List paths under directory; dirs with trailing /. Depth limited."""
        resolved = self._resolve_path(path)
        root = Path(resolved)
        if not root.is_dir():
            return []
        result = []

        def scan(p: Path, depth: int) -> None:
            if depth > max_depth:
                return
            try:
                for child in sorted(p.iterdir()):
                    rel = child.relative_to(root)
                    name = str(rel).replace("\\", "/")
                    if child.is_dir():
                        result.append(name + "/")
                        scan(child, depth + 1)
                    else:
                        result.append(name)
            except PermissionError:
                pass

        scan(root, 0)
        return result


_LOCAL_SANDBOX_ID = "local"


class LocalSandboxProvider(SandboxProvider):
    """
    Provider that returns a single shared LocalSandbox. acquire(thread_id)
    always returns "local"; get("local") returns that instance; release is no-op.
    """

    def __init__(self, path_mappings: Optional[Dict[str, str]] = None):
        self._sandbox = LocalSandbox(path_mappings=path_mappings)

    def acquire(self, thread_id: Optional[str] = None) -> str:
        return _LOCAL_SANDBOX_ID

    def get(self, sandbox_id: str) -> Optional[Sandbox]:
        if sandbox_id == _LOCAL_SANDBOX_ID:
            return self._sandbox
        return None

    def release(self, sandbox_id: str) -> None:
        pass  # no-op for local
