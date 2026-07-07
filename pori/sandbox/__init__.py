"""
Sandbox execution environment for Pori agents.

Provides isolated execution (local or container) with virtual paths
(/mnt/user-data/workspace, uploads, outputs) and thread-scoped lifecycle.
"""

from .base import Sandbox, SandboxProvider, get_sandbox_provider, set_sandbox_provider
from .env_safety import sanitized_subprocess_env
from .local import LocalSandbox, LocalSandboxProvider
from .path_resolution import (
    VIRTUAL_PREFIX,
    ThreadData,
    get_thread_data,
    replace_virtual_path,
    replace_virtual_paths_in_command,
)

__all__ = [
    "Sandbox",
    "SandboxProvider",
    "get_sandbox_provider",
    "set_sandbox_provider",
    "LocalSandbox",
    "LocalSandboxProvider",
    "sanitized_subprocess_env",
    "create_sandbox_provider",
    "VIRTUAL_PREFIX",
    "ThreadData",
    "get_thread_data",
    "replace_virtual_path",
    "replace_virtual_paths_in_command",
]


def create_sandbox_provider(backend: str = "local", **kwargs):
    """Build a sandbox provider by name: "local" (default) or "e2b".

    E2B is a gated capability — the import happens here so the kernel never
    requires the optional dependency unless the backend is actually selected.
    """
    normalized = (backend or "local").strip().lower()
    if normalized == "local":
        return LocalSandboxProvider(**kwargs)
    if normalized == "e2b":
        from .e2b import E2BSandboxProvider

        return E2BSandboxProvider(**kwargs)
    raise ValueError(f"Unknown sandbox backend: {backend!r} (local | e2b)")
