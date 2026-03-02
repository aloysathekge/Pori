"""
Tests for sandbox integration: provider, thread dirs, and bash tool.
"""

import tempfile
from pathlib import Path

import pytest

from pori.sandbox import (
    LocalSandboxProvider,
    set_sandbox_provider,
    get_thread_data,
)
from pori.sandbox.sandbox_tools import bash_tool, BashParams


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
