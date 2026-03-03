"""
Sandbox interface and provider abstraction.

Defines the contract for execution environments (run commands, file I/O)
and for managing sandbox lifecycle per thread/session.
"""

from abc import ABC, abstractmethod
from typing import Optional

# Module-level active provider (singleton per process)
_active_provider: Optional["SandboxProvider"] = None


class Sandbox(ABC):
    """
    Abstract execution environment: run shell commands and do file I/O.
    Implementations may be local (host) or container-based.
    """

    @abstractmethod
    def execute_command(self, command: str) -> str:
        """
        Run a shell command and return combined stdout/stderr as a string.
        """
        ...

    @abstractmethod
    def read_file(self, path: str) -> str:
        """Read file contents as text."""
        ...

    @abstractmethod
    def write_file(self, path: str, content: str, append: bool = False) -> None:
        """Write or append text to a file."""
        ...

    @abstractmethod
    def list_dir(self, path: str, max_depth: int = 2) -> list[str]:
        """
        List paths under directory. Typically dirs have trailing `/`.
        Depth limit avoids unbounded recursion.
        """
        ...


class SandboxProvider(ABC):
    """
    Manages sandbox lifecycle per conversation/thread.
    Same thread_id should get the same sandbox_id across turns.
    """

    @abstractmethod
    def acquire(self, thread_id: Optional[str] = None) -> str:
        """
        Create or reuse a sandbox for the given thread.
        Returns a stable sandbox_id (e.g. "local" or container id).
        """
        ...

    @abstractmethod
    def get(self, sandbox_id: str) -> Optional[Sandbox]:
        """Return the Sandbox instance for the given id, or None."""
        ...

    @abstractmethod
    def release(self, sandbox_id: str) -> None:
        """Tear down the sandbox (e.g. stop container, free resources)."""
        ...


def get_sandbox_provider() -> Optional[SandboxProvider]:
    """Return the currently active sandbox provider, or None."""
    return _active_provider


def set_sandbox_provider(provider: Optional[SandboxProvider]) -> None:
    """Set the active sandbox provider (e.g. from config)."""
    global _active_provider
    _active_provider = provider
