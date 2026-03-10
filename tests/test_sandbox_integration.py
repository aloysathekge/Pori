"""
Tests for sandbox integration: provider, thread dirs, and bash tool.
"""

import tempfile
from pathlib import Path

import pytest

from pori.sandbox import (
    LocalSandboxProvider,
    get_thread_data,
    set_sandbox_provider,
)
from pori.sandbox.path_resolution import VIRTUAL_PREFIX
from pori.sandbox.sandbox_tools import (
    BashParams,
    SandboxListDirParams,
    SandboxReadFileParams,
    SandboxWriteFileParams,
    bash_tool,
    sandbox_list_dir_tool,
    sandbox_read_file_tool,
    sandbox_write_file_tool,
)
from pori.tools.standard.filesystem_tools import (
    CreateDirectoryParams,
    create_directory_tool,
)


@pytest.fixture
def sandbox_env(tmp_path):
    """Enable sandbox with a temp base dir and yield (base_dir, thread_id)."""
    base_dir = str(tmp_path)
    set_sandbox_provider(LocalSandboxProvider())
    yield base_dir, "test_thread_1"
    set_sandbox_provider(None)


def test_get_thread_data_creates_dirs(sandbox_env):
    """Thread data creates workspace, uploads, outputs under base_dir/threads/<id>/user-data/."""
    base_dir, thread_id = sandbox_env
    data = get_thread_data(thread_id, base_dir)
    assert Path(data.workspace_path).is_dir()
    assert Path(data.uploads_path).is_dir()
    assert Path(data.outputs_path).is_dir()
    assert "workspace" in data.workspace_path
    assert "uploads" in data.uploads_path
    assert "outputs" in data.outputs_path
    assert thread_id in data.workspace_path


def test_bash_tool_without_provider_returns_error():
    """Without a sandbox provider, bash tool returns a clear error."""
    set_sandbox_provider(None)
    context = {"thread_id": "x", "sandbox_base_dir": "/tmp"}
    result = bash_tool(BashParams(command="echo hi"), context)
    assert result.get("success") is False
    assert "sandbox" in result.get("error", "").lower()


def test_bash_tool_runs_command(sandbox_env):
    """With provider and context, bash tool runs a command and returns output."""
    base_dir, thread_id = sandbox_env
    context = {"thread_id": thread_id, "sandbox_base_dir": base_dir}
    result = bash_tool(BashParams(command="echo hello from sandbox"), context)
    assert result.get("success") is True
    assert "hello from sandbox" in result.get("output", "")
    assert result.get("sandbox_id") == "local"


def test_bash_tool_creates_thread_dirs_on_first_use(sandbox_env):
    """First bash call with thread_id and sandbox_base_dir creates thread workspace dirs."""
    base_dir, thread_id = sandbox_env
    workspace = Path(base_dir) / "threads" / thread_id / "user-data" / "workspace"
    assert not workspace.exists()
    context = {"thread_id": thread_id, "sandbox_base_dir": base_dir}
    result = bash_tool(BashParams(command="echo ok"), context)
    assert result.get("success") is True
    assert workspace.exists()


def test_sandbox_write_file_and_read_file(sandbox_env):
    """sandbox_write_file and sandbox_read_file use virtual paths and per-thread dirs."""
    base_dir, thread_id = sandbox_env
    context = {"thread_id": thread_id, "sandbox_base_dir": base_dir}
    path = f"{VIRTUAL_PREFIX}/workspace/test_sandbox_file.txt"
    content = "sandbox file content"
    write_result = sandbox_write_file_tool(
        SandboxWriteFileParams(path=path, content=content),
        context,
    )
    assert write_result.get("success") is True
    read_result = sandbox_read_file_tool(SandboxReadFileParams(path=path), context)
    assert read_result.get("success") is True
    assert read_result.get("content") == content
    real_path = (
        Path(base_dir)
        / "threads"
        / thread_id
        / "user-data"
        / "workspace"
        / "test_sandbox_file.txt"
    )
    assert real_path.exists()
    assert real_path.read_text() == content


def test_sandbox_list_dir(sandbox_env):
    """sandbox_list_dir lists entries in the sandbox workspace."""
    base_dir, thread_id = sandbox_env
    context = {"thread_id": thread_id, "sandbox_base_dir": base_dir}
    # Ensure dir exists by writing a file first
    sandbox_write_file_tool(
        SandboxWriteFileParams(path=f"{VIRTUAL_PREFIX}/workspace/foo.txt", content="x"),
        context,
    )
    result = sandbox_list_dir_tool(
        SandboxListDirParams(path=f"{VIRTUAL_PREFIX}/workspace", max_depth=2),
        context,
    )
    assert result.get("success") is True
    entries = result.get("entries", [])
    assert any("foo.txt" in e for e in entries)


def test_create_directory_accepts_sandbox_virtual_path(sandbox_env):
    """Standard create_directory tool accepts /mnt/user-data/... when context has thread_id + sandbox_base_dir."""
    base_dir, thread_id = sandbox_env
    context = {"thread_id": thread_id, "sandbox_base_dir": base_dir}
    result = create_directory_tool(
        CreateDirectoryParams(
            directory_path=f"{VIRTUAL_PREFIX}/workspace/aloy", parents=True
        ),
        context,
    )
    assert result.get("message") is not None, result
    assert "successfully" in result.get("message", "")
    real_dir = (
        Path(base_dir) / "threads" / thread_id / "user-data" / "workspace" / "aloy"
    )
    assert real_dir.is_dir()
