"""
Sandbox execution environment for Pori agents.

Provides isolated execution (local or container) with virtual paths
(/mnt/user-data/workspace, uploads, outputs) and thread-scoped lifecycle.
"""

from .base import Sandbox, SandboxProvider, get_sandbox_provider, set_sandbox_provider
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
    "VIRTUAL_PREFIX",
    "ThreadData",
    "get_thread_data",
    "replace_virtual_path",
    "replace_virtual_paths_in_command",
]
