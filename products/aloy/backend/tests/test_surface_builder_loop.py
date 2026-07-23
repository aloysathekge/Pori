import json
from pathlib import Path

import pytest

from aloy_backend.surface_build_runner import SurfaceBuildRunnerResult
from aloy_backend.surface_builder_loop import run_surface_builder_loop
from aloy_backend.surface_development_workspace import LocalGitSurfaceWorkspace
from pori import SystemMessage, UserMessage
from pori.llm.messages import ToolCall, ToolResultMessage, ToolTurn


class _Runner:
    toolchain_version = "test@1"

    def __init__(self) -> None:
        self.calls = 0

    async def build(self, *, build_id, files, manifest):
        del build_id, manifest
        self.calls += 1
        content = files.get("/src/App.tsx", "")
        return SurfaceBuildRunnerResult(
            status="failed" if "BROKEN" in content else "succeeded",
            bundle=None if "BROKEN" in content else b"bundle",
            diagnostics=(
                [{"code": "typescript_contract_error", "message": "Fix BROKEN"}]
                if "BROKEN" in content
                else []
            ),
        )


def _files(value: str = "Old") -> dict[str, str]:
    return {
        "/surface.json": (
            '{"sdk_version":"1","entrypoint":"/src/App.tsx",'
            '"capabilities":[],"intents":{},"widgets":[],"interaction_checks":[],'
            '"primary_jobs":[]}'
        ),
        "/src/App.tsx": f"export default function App() {{ return <main>{value}</main>; }}",
    }


class _NativeModel:
    def __init__(self) -> None:
        self.invocations = 0
        self.turns = [
            ToolTurn(
                tool_calls=[
                    ToolCall(
                        id="one",
                        name="replace_text",
                        arguments={
                            "path": "/src/App.tsx",
                            "match": "Old",
                            "replacement": "New",
                        },
                    ),
                    ToolCall(id="two", name="run_typecheck", arguments={}),
                ]
            ),
            ToolTurn(
                tool_calls=[
                    ToolCall(
                        id="three",
                        name="finish_candidate",
                        arguments={"summary": "Update the view"},
                    )
                ]
            ),
        ]

    async def ainvoke_tools(self, messages, tools):
        assert tools
        assert messages
        if self.invocations == 1:
            assert any(isinstance(message, ToolResultMessage) for message in messages)
        self.invocations += 1
        return self.turns.pop(0)


class _TextModel:
    def __init__(self) -> None:
        self.invocations = 0
        self.turns = [
            '{"actions":[{"name":"replace_text","arguments":{"path":"/src/App.tsx",'
            '"match":"BROKEN","replacement":"Working"}}]}',
            '{"actions":[{"name":"finish_candidate","arguments":{"summary":"Repair"}}]}',
        ]

    async def ainvoke(self, messages):
        assert messages
        assert not any(isinstance(message, ToolResultMessage) for message in messages)
        if self.invocations == 0:
            assert '"name": "read_file"' in messages[0].content
        self.invocations += 1
        return self.turns.pop(0)


class _NativeTextFallbackModel:
    def __init__(self) -> None:
        self.turns = [
            ToolTurn(
                text='{"actions":[{"name":"replace_text","arguments":{'
                '"path":"/src/App.tsx","match":"Old","replacement":"Fallback"}}]}'
            ),
            ToolTurn(
                text='{"actions":[{"name":"finish_candidate","arguments":{'
                '"summary":"Use the fallback"}}]}'
            ),
        ]

    async def ainvoke_tools(self, messages, tools):
        assert tools
        assert not any(isinstance(message, ToolResultMessage) for message in messages)
        if len(self.turns) == 1:
            assert any(
                isinstance(message, UserMessage)
                and "Trusted workspace results" in message.content
                for message in messages
            )
        return self.turns.pop(0)


@pytest.mark.asyncio
async def test_native_builder_edits_checks_and_finishes_one_git_candidate(
    tmp_path: Path,
):
    runner = _Runner()
    workspace = await LocalGitSurfaceWorkspace.create(
        workspace_id="native",
        base_files=_files(),
        build_runner=runner,
        parent=tmp_path,
    )
    try:
        result = await run_surface_builder_loop(
            llm=_NativeModel(),
            workspace=workspace,
            messages=[SystemMessage(content="Build"), UserMessage(content="Change it")],
            primary_jobs=["Use the updated view"],
            capabilities={"tools"},
        )
        assert result.protocol == "native_tools"
        assert result.turns == 2
        assert result.tool_calls == 3
        assert runner.calls == 1
        assert "New" in await workspace.read_file("/src/App.tsx")
        assert result.finished.receipt.changed_paths == ["src/App.tsx"]
    finally:
        await workspace.close()


@pytest.mark.asyncio
async def test_text_action_protocol_repairs_without_structured_output(tmp_path: Path):
    runner = _Runner()
    workspace = await LocalGitSurfaceWorkspace.create(
        workspace_id="text",
        base_files=_files("BROKEN"),
        build_runner=runner,
        parent=tmp_path,
    )
    try:
        result = await run_surface_builder_loop(
            llm=_TextModel(),
            workspace=workspace,
            messages=[SystemMessage(content="Build"), UserMessage(content="Repair it")],
            primary_jobs=["Use the repaired view"],
            capabilities=set(),
        )
        assert result.protocol == "action_json"
        assert result.turns == 2
        assert runner.calls == 1
        assert "Working" in await workspace.read_file("/src/App.tsx")
    finally:
        await workspace.close()


_REQUIRED_JOB = {"id": "job_" + "a" * 16, "description": "See the revised view"}


def _stale_contract_files() -> dict[str, str]:
    """Source published by an earlier request, declaring that request's jobs."""
    stale_jobs = [
        {
            "id": "job_" + "b" * 16,
            "description": "Track applications",
            "assertions": [
                {"kind": "visible", "role": "heading", "name": "Applications"}
            ],
        },
        {
            "id": "job_" + "c" * 16,
            "description": "Record an interview",
            "assertions": [{"kind": "visible", "role": "button", "name": "Record"}],
        },
    ]
    files = _files()
    manifest = (
        '{"format":"aloy-react-surface","entrypoint":"/src/App.tsx",'
        '"sdk_version":"1","capabilities":[],"intents":{},"widgets":[],'
        '"interaction_checks":[],"primary_jobs":' + json.dumps(stale_jobs) + "}"
    )
    return {**files, "/surface.json": manifest}


class _RevisionContractModel:
    """Finishes against the stale manifest first, then repairs it in-workspace."""

    def __init__(self) -> None:
        corrected = [
            {
                **_REQUIRED_JOB,
                "assertions": [
                    {"kind": "visible", "role": "heading", "name": "Revised"}
                ],
            }
        ]
        self.turns = [
            ToolTurn(
                tool_calls=[
                    ToolCall(
                        id="edit",
                        name="replace_text",
                        arguments={
                            "path": "/src/App.tsx",
                            "match": "Old",
                            "replacement": "New",
                        },
                    ),
                    ToolCall(id="check", name="run_typecheck", arguments={}),
                    ToolCall(
                        id="finish-stale",
                        name="finish_candidate",
                        arguments={"summary": "Revise the view"},
                    ),
                ]
            ),
            ToolTurn(
                tool_calls=[
                    ToolCall(
                        id="manifest",
                        name="write_file",
                        arguments={
                            "path": "/surface.json",
                            "content": (
                                '{"format":"aloy-react-surface",'
                                '"entrypoint":"/src/App.tsx","sdk_version":"1",'
                                '"capabilities":[],"intents":{},"widgets":[],'
                                '"interaction_checks":[],"primary_jobs":'
                                + json.dumps(corrected)
                                + "}"
                            ),
                        },
                    ),
                    ToolCall(id="recheck", name="run_typecheck", arguments={}),
                    ToolCall(
                        id="finish-fixed",
                        name="finish_candidate",
                        arguments={"summary": "Revise the view"},
                    ),
                ]
            ),
        ]
        self.refusals: list[dict] = []

    async def ainvoke_tools(self, messages, tools):
        assert tools
        for message in messages:
            if isinstance(message, ToolResultMessage):
                payload = json.loads(message.content)
                result = payload.get("result") or {}
                if result.get("reason") == "primary_job_contract_failed":
                    self.refusals.append(result)
        return self.turns.pop(0)


@pytest.mark.asyncio
async def test_finish_refuses_stale_primary_job_contract_before_paid_submission(
    tmp_path: Path,
):
    """A revision inherits the old manifest jobs; finish must mirror the host
    gate and refuse in-workspace instead of letting the paid host submission
    fail deterministically."""
    model = _RevisionContractModel()
    workspace = await LocalGitSurfaceWorkspace.create(
        workspace_id="revision-contract",
        base_files=_stale_contract_files(),
        build_runner=_Runner(),
        parent=tmp_path,
    )
    try:
        result = await run_surface_builder_loop(
            llm=model,
            workspace=workspace,
            messages=[SystemMessage(content="Build"), UserMessage(content="Revise it")],
            primary_jobs=[_REQUIRED_JOB["description"]],
            capabilities={"tools"},
            required_primary_jobs=[_REQUIRED_JOB],
        )
        assert model.refusals, "stale manifest must be refused at finish"
        assert any(
            diagnostic["code"] == "primary_job_manifest_mismatch"
            for refusal in model.refusals
            for diagnostic in refusal["diagnostics"]
        )
        declared = json.loads(
            {
                item.source_path: item.content
                for item in result.finished.candidate.files
            }["/surface.json"]
        )["primary_jobs"]
        assert [
            {"id": item["id"], "description": item["description"]} for item in declared
        ] == [_REQUIRED_JOB]
        assert "New" in await workspace.read_file("/src/App.tsx")
    finally:
        await workspace.close()


@pytest.mark.asyncio
async def test_native_provider_can_fall_back_to_text_actions_without_fake_tool_results(
    tmp_path: Path,
):
    workspace = await LocalGitSurfaceWorkspace.create(
        workspace_id="native-text-fallback",
        base_files=_files(),
        build_runner=_Runner(),
        parent=tmp_path,
    )
    try:
        result = await run_surface_builder_loop(
            llm=_NativeTextFallbackModel(),
            workspace=workspace,
            messages=[SystemMessage(content="Build"), UserMessage(content="Change it")],
            primary_jobs=["Use the fallback view"],
            capabilities={"tools"},
        )
        assert result.protocol == "action_json"
        assert "Fallback" in await workspace.read_file("/src/App.tsx")
    finally:
        await workspace.close()
