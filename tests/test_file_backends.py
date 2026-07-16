from __future__ import annotations

from typing import Dict, Optional

import pytest

from pori import (
    FILE_BACKEND_CONTEXT_KEY,
    AgentSettings,
    CompositeFileBackend,
    FileErrorCode,
    FileMount,
    MemoryFileBackend,
    Orchestrator,
    ReadOnlyFileBackend,
    SandboxFileBackend,
    normalize_virtual_path,
)
from pori.sandbox.base import Sandbox
from pori.tools.standard.filesystem_tools import (
    EditFileParams,
    ListDirectoryParams,
    ReadFileParams,
    WriteFileParams,
    edit_file_tool,
    list_directory_tool,
    read_file_tool,
    write_file_tool,
)


def test_virtual_paths_are_canonical_and_traversal_is_rejected():
    assert (
        normalize_virtual_path("//workspace/./src/app.tsx") == "/workspace/src/app.tsx"
    )
    with pytest.raises(ValueError, match="traversal"):
        normalize_virtual_path("/workspace/../secrets")
    with pytest.raises(ValueError, match="backslashes"):
        normalize_virtual_path("/workspace\\secrets")


def test_memory_backend_supports_paginated_reads_and_exact_edits():
    backend = MemoryFileBackend({"/notes.md": "one\ntwo\nthree\n"})

    page = backend.read("/notes.md", offset=1, limit=1)
    assert page.content == "two"
    assert page.total_lines == 3
    assert page.start_line == 2
    assert page.end_line == 2
    assert page.next_offset == 2

    edited = backend.edit("/notes.md", "two", "TWO")
    assert edited.success
    assert edited.replacements == 1
    assert backend.read("/notes.md").content == "one\nTWO\nthree\n"


def test_composite_backend_uses_longest_prefix_and_denies_unmounted_paths():
    workspace = MemoryFileBackend()
    vendor = MemoryFileBackend()
    backend = CompositeFileBackend(
        (
            FileMount("/workspace", workspace),
            FileMount("/workspace/vendor", vendor),
        )
    )

    assert backend.write("/workspace/app.tsx", "app").success
    assert backend.write("/workspace/vendor/pkg.ts", "vendor").success
    assert workspace.read("/app.tsx").content == "app"
    assert vendor.read("/pkg.ts").content == "vendor"

    assert [entry.path for entry in backend.list("/").entries] == ["/workspace"]
    assert [entry.path for entry in backend.list("/workspace").entries] == [
        "/workspace/app.tsx",
        "/workspace/vendor",
    ]

    missing = backend.read("/event/private.md")
    assert missing.error_code is FileErrorCode.PERMISSION_DENIED
    assert "No virtual file mount" in (missing.error or "")


def test_read_only_mount_denies_mutation_but_remains_readable():
    event = ReadOnlyFileBackend(
        MemoryFileBackend({"/brief.md": "Official Event context"})
    )
    backend = CompositeFileBackend((FileMount("/event", event, read_only=True),))

    assert backend.read("/event/brief.md").content == "Official Event context"
    denied = backend.write("/event/brief.md", "changed")
    assert denied.error_code is FileErrorCode.PERMISSION_DENIED


class _FakeSandbox(Sandbox):
    def __init__(self):
        self.files: Dict[str, str] = {}

    def execute_command(self, command: str, cwd: Optional[str] = None) -> str:
        raise AssertionError("File backend must never execute shell commands")

    def read_file(self, path: str) -> str:
        return self.files[path]

    def write_file(self, path: str, content: str, append: bool = False) -> None:
        previous = self.files.get(path, "") if append else ""
        self.files[path] = previous + content

    def list_dir(self, path: str, max_depth: int = 2) -> list[str]:
        prefix = path.rstrip("/") + "/"
        return sorted(
            key[len(prefix) :] for key in self.files if key.startswith(prefix)
        )


def test_sandbox_adapter_is_file_only_and_root_scoped():
    sandbox = _FakeSandbox()
    backend = SandboxFileBackend(sandbox, root="/surface")

    assert backend.write("/src/App.tsx", "export default 1").success
    assert sandbox.files["/surface/src/App.tsx"] == "export default 1"
    assert backend.read("/src/App.tsx").content == "export default 1"
    assert [entry.path for entry in backend.list("/src").entries] == ["/src/App.tsx"]
    assert backend.read("/../escape").error_code is FileErrorCode.INVALID_PATH


def test_standard_file_tools_route_virtual_paths_without_host_fallback():
    event = MemoryFileBackend({"/brief.md": "Read only\nSecond line"})
    workspace = MemoryFileBackend()
    backend = CompositeFileBackend(
        (
            FileMount("/event", event, read_only=True),
            FileMount("/workspace", workspace),
        )
    )
    context = {FILE_BACKEND_CONTEXT_KEY: backend}

    read = read_file_tool(
        ReadFileParams(file_path="/event/brief.md", offset=1, max_lines=1),
        context,
    )
    assert read["success"] is True
    assert read["content"] == "Second line"

    written = write_file_tool(
        WriteFileParams(file_path="/workspace/App.tsx", content="const x = 1"),
        context,
    )
    assert written["success"] is True
    assert workspace.read("/App.tsx").content == "const x = 1"

    edited = edit_file_tool(
        EditFileParams(
            file_path="/workspace/App.tsx",
            old_string="1",
            new_string="2",
        ),
        context,
    )
    assert edited["success"] is True
    assert workspace.read("/App.tsx").content == "const x = 2"

    listed = list_directory_tool(
        ListDirectoryParams(directory_path="/workspace"),
        context,
    )
    assert listed["success"] is True
    assert [item["path"] for item in listed["items"]] == ["/workspace/App.tsx"]

    denied = write_file_tool(
        WriteFileParams(file_path="/event/brief.md", content="tamper"),
        context,
    )
    assert denied["success"] is False
    assert denied["error_code"] is FileErrorCode.PERMISSION_DENIED


@pytest.mark.asyncio
async def test_orchestrator_injects_one_scoped_backend(mock_llm, tool_registry):
    backend = CompositeFileBackend((FileMount("/workspace", MemoryFileBackend()),))
    orchestrator = Orchestrator(
        llm=mock_llm,
        tools_registry=tool_registry,
        file_backend=backend,
    )

    result = await orchestrator.execute_task(
        "Inspect the workspace",
        agent_settings=AgentSettings(max_steps=1),
    )

    assert result["agent"]._tool_context_extra[FILE_BACKEND_CONTEXT_KEY] is backend
