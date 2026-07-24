"""Builder eval harness mechanics, proven with scripted models (no spend)."""

from __future__ import annotations

import json
import shutil

import pytest

from aloy_backend.builder_eval import CASES, EvalCase, run_case
from aloy_backend.surface_build_runner import SurfaceBuildRunnerResult
from pori.llm.messages import ToolCall, ToolTurn

pytestmark = pytest.mark.asyncio


class _FakeRunner:
    """Compile-free build runner so mechanics tests stay fast."""

    toolchain_version = "eval-test@1"

    async def build(self, *, build_id, files, manifest):
        del build_id, files, manifest
        return SurfaceBuildRunnerResult(status="succeeded", bundle=b"bundle")


def _case(required=None) -> EvalCase:
    jobs = required or [{"id": "job_" + "e" * 16, "description": "See the eval view"}]
    manifest = {
        "format": "aloy-react-surface",
        "entrypoint": "/src/App.tsx",
        "sdk_version": "1",
        "capabilities": [],
        "intents": {},
        "widgets": [],
        "interaction_checks": [],
        "primary_jobs": [
            {
                **job,
                "assertions": [{"kind": "visible", "role": "heading", "name": "Eval"}],
            }
            for job in jobs
        ],
    }
    return EvalCase(
        name="scripted",
        task="Change the view",
        required_primary_jobs=jobs,
        base_files={
            "/surface.json": json.dumps(manifest),
            "/src/App.tsx": "export default function App(){return <main>Old</main>}",
        },
    )


class _ScriptedModel:
    model = "scripted"
    last_usage = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}

    def __init__(self) -> None:
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
                        id="finish",
                        name="finish_candidate",
                        arguments={"summary": "Scripted change"},
                    ),
                ]
            )
        ]

    async def ainvoke_tools(self, messages, tools):
        del messages, tools
        return self.turns.pop(0)


async def test_scripted_pass_reports_metrics(monkeypatch):
    import aloy_backend.builder_eval as eval_module

    async def fake_gate(candidate, *, required_primary_jobs, build_runner):
        del candidate, required_primary_jobs, build_runner
        return []

    monkeypatch.setattr(eval_module, "_full_gate", fake_gate)
    result = await run_case(
        _case(),
        llm_factory=_ScriptedModel,
        build_runner=_FakeRunner(),
    )
    assert result["status"] == "published"
    assert result["gate_attempts"] == 1
    assert result["turns"] == 1
    assert result["llm_calls"] == 1
    assert result["estimated_cost_usd"] >= 0


async def test_gate_rejection_feeds_repair_and_detects_no_progress(monkeypatch):
    import aloy_backend.builder_eval as eval_module

    async def failing_gate(candidate, *, required_primary_jobs, build_runner):
        del candidate, required_primary_jobs, build_runner
        return [{"stage": "preview", "code": "x", "message": "rejected"}]

    monkeypatch.setattr(eval_module, "_full_gate", failing_gate)

    class RepeatingModel(_ScriptedModel):
        def __init__(self) -> None:
            super().__init__()
            first = self.turns[0]
            # Second attempt edits back and forth so source changes, third
            # attempt is identical to the second -> no_progress detection.
            self.turns = [
                first,
                ToolTurn(
                    tool_calls=[
                        ToolCall(
                            id="e2",
                            name="replace_text",
                            arguments={
                                "path": "/src/App.tsx",
                                "match": "New",
                                "replacement": "Newer",
                            },
                        ),
                        ToolCall(id="c2", name="run_typecheck", arguments={}),
                        ToolCall(
                            id="f2",
                            name="finish_candidate",
                            arguments={"summary": "Second"},
                        ),
                    ]
                ),
                ToolTurn(
                    tool_calls=[
                        ToolCall(
                            id="e3",
                            name="replace_text",
                            arguments={
                                "path": "/src/App.tsx",
                                "match": "Newer",
                                "replacement": "Newer2",
                            },
                        ),
                        ToolCall(id="c3", name="run_typecheck", arguments={}),
                        ToolCall(
                            id="f3",
                            name="finish_candidate",
                            arguments={"summary": "Third"},
                        ),
                    ]
                ),
            ]

    result = await run_case(
        _case(),
        llm_factory=RepeatingModel,
        build_runner=_FakeRunner(),
    )
    assert result["status"] == "gate_failed"
    assert result["gate_attempts"] == 3
    assert result["failure"][0]["code"] == "x"


async def test_builtin_cases_use_the_baseline_and_real_gate():
    if shutil.which("node") is None:
        pytest.skip("Node.js is not installed")
    case = CASES["revision-small"]()
    assert "/src/App.tsx" in case.base_files
    assert "/surface.json" in case.base_files
    # A scripted model that rewrites surface.json to the required contract and
    # adds the Notes view, then finishes: the REAL gate must pass it.
    manifest = json.loads(case.base_files["/surface.json"])
    manifest["primary_jobs"] = [
        {
            "id": case.required_primary_jobs[0]["id"],
            "description": case.required_primary_jobs[0]["description"],
            "assertions": [{"kind": "visible", "role": "heading", "name": "Overview"}],
        },
        {
            "id": case.required_primary_jobs[1]["id"],
            "description": case.required_primary_jobs[1]["description"],
            "steps": [{"action": "click", "role": "button", "name": "Notes"}],
            "assertions": [{"kind": "visible", "role": "heading", "name": "Notes"}],
        },
    ]

    notes_view = (
        "import { useState } from 'react';\n"
        "import { Section } from '../primitives';\n"
        "export function NotesView() {\n"
        "  const [note, setNote] = useState('');\n"
        "  const [notes, setNotes] = useState<string[]>([]);\n"
        "  return (\n"
        '    <Section heading="Notes">\n'
        '      <form className="baseline-ask" onSubmit={(event) => {\n'
        "        event.preventDefault();\n"
        "        if (!note.trim()) return;\n"
        "        setNotes([...notes, note.trim()]);\n"
        "        setNote('');\n"
        "      }}>\n"
        '        <label><span className="baseline-label">New note</span>\n'
        '          <input aria-label="New note" value={note}\n'
        "            onChange={(event) => setNote(event.target.value)} />\n"
        "        </label>\n"
        '        <button type="submit">Add note</button>\n'
        "      </form>\n"
        '      <ul className="baseline-list">\n'
        "        {notes.map((item, index) => (\n"
        '          <li key={index}><span className="baseline-list-title">{item}</span></li>\n'
        "        ))}\n"
        "      </ul>\n"
        "    </Section>\n"
        "  );\n"
        "}\n"
    )

    class BaselineEditor:
        model = "scripted"
        last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        def __init__(self) -> None:
            app = case.base_files["/src/App.tsx"]
            app = app.replace(
                "import { Overview } from './views/Overview';",
                "import { NotesView } from './views/Notes';\n"
                "import { Overview } from './views/Overview';",
            )
            app = app.replace(
                "type ViewName = 'overview' | 'tasks' | 'files';",
                "type ViewName = 'overview' | 'tasks' | 'files' | 'notes';",
            )
            app = app.replace(
                """        <button
          type="button"
          aria-pressed={view === 'files'}
          onClick={() => setView('files')}
        >
          Files
        </button>""",
                """        <button
          type="button"
          aria-pressed={view === 'files'}
          onClick={() => setView('files')}
        >
          Files
        </button>
        <button
          type="button"
          aria-pressed={view === 'notes'}
          onClick={() => setView('notes')}
        >
          Notes
        </button>""",
            )
            app = app.replace(
                """      ) : view === 'tasks' ? (
        <TasksView />
      ) : (
        <FilesView />
      )}""",
                """      ) : view === 'tasks' ? (
        <TasksView />
      ) : view === 'notes' ? (
        <NotesView />
      ) : (
        <FilesView />
      )}""",
            )
            self.turns = [
                ToolTurn(
                    tool_calls=[
                        ToolCall(
                            id="manifest",
                            name="write_file",
                            arguments={
                                "path": "/surface.json",
                                "content": json.dumps(manifest),
                            },
                        ),
                        ToolCall(
                            id="notes",
                            name="write_file",
                            arguments={
                                "path": "/src/views/Notes.tsx",
                                "content": notes_view,
                            },
                        ),
                        ToolCall(
                            id="app",
                            name="write_file",
                            arguments={"path": "/src/App.tsx", "content": app},
                        ),
                        ToolCall(id="check", name="run_typecheck", arguments={}),
                        ToolCall(
                            id="finish",
                            name="finish_candidate",
                            arguments={"summary": "Add a Notes view"},
                        ),
                    ]
                )
            ]

        async def ainvoke_tools(self, messages, tools):
            del messages, tools
            return self.turns.pop(0)

    result = await run_case(case, llm_factory=BaselineEditor)
    assert result["status"] == "published", result["failure"]
    assert result["gate_attempts"] == 1
