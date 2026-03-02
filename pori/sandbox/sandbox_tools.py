"""
Sandbox-backed tools: run commands and file I/O through the active sandbox.

Context should include (when using sandbox):
  - thread_id: str — used to acquire/reuse sandbox and thread dirs
  - thread_data: ThreadData — workspace_path, uploads_path, outputs_path (for path resolution)
  - sandbox_id: str (optional) — set after first acquire
  - sandbox_base_dir: str (optional) — base dir for thread data; used to build thread_data if missing

If no sandbox provider is set, the bash tool returns an error asking to enable sandbox.
"""

from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from pori.tools.registry import tool_registry

Registry = tool_registry()

try:
    from .base import get_sandbox_provider
    from .path_resolution import (
        VIRTUAL_PREFIX,
        get_thread_data,
        replace_virtual_path,
        replace_virtual_paths_in_command,
        ThreadData,
    )
except ImportError:
    get_sandbox_provider = None
    get_thread_data = None
    replace_virtual_path = None
    replace_virtual_paths_in_command = None
    ThreadData = None
    VIRTUAL_PREFIX = "/mnt/user-data"


class BashParams(BaseModel):
    command: str = Field(..., description="Shell command to run (e.g. bash -c 'echo hello')")
    working_dir: Optional[str] = Field(
        None,
        description=f"Working directory. Use {VIRTUAL_PREFIX}/workspace for scratch, {VIRTUAL_PREFIX}/outputs for results.",
    )


def _ensure_sandbox_and_thread_dirs(context: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str], Optional[Any], str]:
    """
    Ensure sandbox is acquired and thread dirs exist (for local).
    Returns (sandbox, sandbox_id, thread_data, error_message).
    If error_message is non-empty, sandbox may be None.
    """
    if get_sandbox_provider is None:
        return None, None, None, "Sandbox module not available."
    provider = get_sandbox_provider()
    if provider is None:
        return None, None, None, "No sandbox provider configured. Set one via set_sandbox_provider(LocalSandboxProvider())."
    thread_id = (context or {}).get("thread_id") or (context or {}).get("task_id") or "default"
    thread_data = (context or {}).get("thread_data")
    sandbox_id = (context or {}).get("sandbox_id")
    if sandbox_id is None:
        sandbox_id = provider.acquire(thread_id)
        if context is not None:
            context["sandbox_id"] = sandbox_id
    sandbox = provider.get(sandbox_id)
    if sandbox is None:
        return None, None, None, f"Sandbox not found for id: {sandbox_id}"
    # Ensure thread dirs for local (when we have base_dir and get_thread_data)
    if thread_data is None and get_thread_data is not None:
        base_dir = (context or {}).get("sandbox_base_dir")
        if base_dir:
            thread_data = get_thread_data(thread_id, base_dir)
            thread_data.ensure_dirs()
            if context is not None:
                context["thread_data"] = thread_data
    return sandbox, sandbox_id, thread_data, ""


@Registry.tool(
    name="bash",
    param_model=BashParams,
    description=f"Run a shell command in the sandbox. Prefer paths under {VIRTUAL_PREFIX}/workspace, {VIRTUAL_PREFIX}/uploads, {VIRTUAL_PREFIX}/outputs.",
)
def bash_tool(params: BashParams, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a shell command via the active sandbox. For local sandbox, virtual paths in the command are rewritten to thread dirs."""
    sandbox, sandbox_id, thread_data, err = _ensure_sandbox_and_thread_dirs(context)
    if err:
        return {"success": False, "error": err}
    command = params.command
    if sandbox_id == "local" and thread_data is not None and replace_virtual_paths_in_command is not None:
        command = replace_virtual_paths_in_command(command, thread_data)
    try:
        output = sandbox.execute_command(command)
        return {"success": True, "output": output, "sandbox_id": sandbox_id}
    except Exception as e:
        return {"success": False, "error": str(e)}
