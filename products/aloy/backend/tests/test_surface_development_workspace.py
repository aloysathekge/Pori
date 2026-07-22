from pathlib import Path

import pytest

from aloy_backend.surface_build_runner import SurfaceBuildRunnerResult
from aloy_backend.surface_development_workspace import (
    LocalGitSurfaceWorkspace,
    SurfaceWorkspaceConflictError,
    SurfaceWorkspaceEdit,
    SurfaceWorkspaceError,
)


class _BuildRunner:
    toolchain_version = "test@1"

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def build(self, *, build_id, files, manifest):
        del build_id, manifest
        self.calls.append(dict(files))
        if "BROKEN" in files.get("/src/App.tsx", ""):
            return SurfaceBuildRunnerResult(
                status="failed",
                diagnostics=[
                    {
                        "code": "typescript_contract_error",
                        "message": "BROKEN is not valid",
                        "path": "/src/App.tsx",
                    }
                ],
            )
        return SurfaceBuildRunnerResult(
            status="succeeded",
            bundle=b"bundle",
            resource_metrics={"compile_ms": 12.0},
        )


def _files(title: str = "University") -> dict[str, str]:
    return {
        "/surface.json": (
            '{"sdk_version":"1","entrypoint":"/src/App.tsx",'
            '"capabilities":[],"intents":{},"widgets":[],"interaction_checks":[],'
            '"primary_jobs":[]}'
        ),
        "/src/App.tsx": f"export default function App() {{ return <main>{title}</main>; }}",
    }


@pytest.mark.asyncio
async def test_local_workspace_tracks_exact_diff_and_finishes_checked_source(
    tmp_path: Path,
):
    runner = _BuildRunner()
    workspace = await LocalGitSurfaceWorkspace.create(
        workspace_id="run_123",
        base_files=_files(),
        build_runner=runner,
        parent=tmp_path,
    )
    try:
        assert await workspace.list_files() == ["/src/App.tsx", "/surface.json"]
        assert "University" in await workspace.read_file("/src/App.tsx")
        assert (await workspace.search_source("university"))[0]["line"] == 1

        changed = await workspace.apply(
            [
                SurfaceWorkspaceEdit(
                    operation="replace_text",
                    path="/src/App.tsx",
                    match="University",
                    replacement="Madrid",
                )
            ]
        )
        assert changed["changed_paths"] == ["/src/App.tsx"]
        check = await workspace.check()
        assert check.status == "succeeded"
        finished = await workspace.finish(
            summary="Rename the view",
            primary_jobs=["See the renamed view"],
        )
        assert finished.receipt.base_commit != finished.receipt.candidate_commit
        assert finished.receipt.changed_paths == ["src/App.tsx"]
        assert "Madrid" in finished.candidate.files[0].content
        assert runner.calls[-1]["/src/App.tsx"].find("Madrid") > 0
    finally:
        await workspace.close()


@pytest.mark.asyncio
async def test_workspace_requires_a_fresh_successful_check(tmp_path: Path):
    workspace = await LocalGitSurfaceWorkspace.create(
        workspace_id="run_456",
        base_files=_files(),
        build_runner=_BuildRunner(),
        parent=tmp_path,
    )
    try:
        await workspace.apply(
            [
                SurfaceWorkspaceEdit(
                    operation="replace_text",
                    path="/src/App.tsx",
                    match="University",
                    replacement="BROKEN",
                )
            ]
        )
        assert (await workspace.check()).status == "failed"
        with pytest.raises(SurfaceWorkspaceConflictError):
            await workspace.finish(summary="Broken", primary_jobs=[])
        await workspace.apply(
            [
                SurfaceWorkspaceEdit(
                    operation="replace_text",
                    path="/src/App.tsx",
                    match="BROKEN",
                    replacement="Working",
                )
            ]
        )
        with pytest.raises(SurfaceWorkspaceConflictError):
            await workspace.finish(summary="Unchecked", primary_jobs=[])
    finally:
        await workspace.close()


@pytest.mark.asyncio
async def test_workspace_rejects_non_atomic_or_unsafe_edits(tmp_path: Path):
    workspace = await LocalGitSurfaceWorkspace.create(
        workspace_id="run_789",
        base_files=_files(),
        build_runner=_BuildRunner(),
        parent=tmp_path,
    )
    try:
        with pytest.raises(SurfaceWorkspaceError, match="exactly once"):
            await workspace.apply(
                [
                    SurfaceWorkspaceEdit(
                        operation="replace_text",
                        path="/src/App.tsx",
                        match="missing",
                        replacement="value",
                    )
                ]
            )
        assert "University" in await workspace.read_file("/src/App.tsx")
        with pytest.raises(ValueError):
            SurfaceWorkspaceEdit(
                operation="write",
                path="/workspace/../secret.txt",
                content="no",
            )
    finally:
        await workspace.close()
