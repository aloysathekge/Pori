"""Extraction OUT: run artifacts move from the thread dirs into the object
store with pointer rows, jailed paths only, caps honored, no silent drops."""

from types import SimpleNamespace

import pytest

import aloy_backend.run_outcome as run_outcome_mod
import aloy_backend.storage as storage_mod
from aloy_backend.models import EventTrailEntry, StoredFile
from aloy_backend.run_outcome import RunOutcome, store_run_artifacts
from pori import get_workspace_data


class _Session:
    """Only session.add is used by store_run_artifacts."""

    def __init__(self):
        self.added = []

    def add(self, row):
        self.added.append(row)


@pytest.fixture
def env(tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox"
    storage = tmp_path / "storage"
    monkeypatch.setattr(
        run_outcome_mod.settings, "sandbox_base_dir", str(sandbox), raising=False
    )
    monkeypatch.setattr(
        storage_mod.settings, "storage_dir", str(storage), raising=False
    )
    monkeypatch.setattr(storage_mod, "_STORE", None)  # rebuild against tmp dir
    thread = get_workspace_data("evt-artifacts", "run-artifacts", str(sandbox))
    conv = SimpleNamespace(id="conv-1", event_id="evt-artifacts", agent_config_id=None)
    ctx = SimpleNamespace(organization_id="org-1", user_id="user-1")
    return SimpleNamespace(thread=thread, conv=conv, ctx=ctx)


def _outcome(artifacts):
    return RunOutcome(
        task="t",
        final_answer="a",
        reasoning=None,
        success=True,
        steps_taken=1,
        metrics=None,
        trace=None,
        artifacts=artifacts,
        run_id="run-artifacts",
        event_id="evt-artifacts",
        session_id="conv-1",
        organization_id="org-1",
    )


def test_virtual_and_relative_paths_are_stored(env):
    with open(env.thread.outputs_path + "/chart.png", "wb") as f:
        f.write(b"PNG")
    with open(env.thread.workspace_path + "/notes.md", "w") as f:
        f.write("hi")

    artifacts = [
        {"kind": "file", "path": "/mnt/user-data/outputs/chart.png"},
        {"kind": "file", "path": "notes.md"},
    ]
    session = _Session()
    store_run_artifacts(session, env.conv, env.ctx, _outcome(artifacts))

    rows = [r for r in session.added if isinstance(r, StoredFile)]
    trail = [r for r in session.added if isinstance(r, EventTrailEntry)]
    assert len(rows) == 2
    assert len(trail) == 2
    assert all(a.get("file_id") for a in artifacts)
    assert {r.name for r in rows} == {"chart.png", "notes.md"}
    assert all(r.organization_id == "org-1" and r.run_id for r in rows)
    # Bytes actually landed in the store.
    store = storage_mod.get_object_store()
    with store.open(rows[0].storage_key) as fh:
        assert fh.read() in (b"PNG", b"hi")


def test_paths_outside_the_jail_are_never_uploaded(env, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("leak")
    artifacts = [
        {"kind": "file", "path": str(secret)},
        {"kind": "file", "path": "../../../../etc/passwd"},
        {"kind": "file", "path": "(path unavailable)"},
    ]
    session = _Session()
    store_run_artifacts(session, env.conv, env.ctx, _outcome(artifacts))
    assert not [r for r in session.added if isinstance(r, StoredFile)]
    assert not any(a.get("file_id") for a in artifacts)


def test_oversize_file_is_marked_not_silently_dropped(env, monkeypatch):
    monkeypatch.setattr(
        run_outcome_mod.settings, "storage_max_artifact_mb", 0, raising=False
    )
    with open(env.thread.outputs_path + "/big.bin", "wb") as f:
        f.write(b"x" * 10)
    artifacts = [{"kind": "file", "path": "/mnt/user-data/outputs/big.bin"}]
    session = _Session()
    store_run_artifacts(session, env.conv, env.ctx, _outcome(artifacts))
    assert artifacts[0].get("too_large") is True
    assert not [r for r in session.added if isinstance(r, StoredFile)]


def test_same_file_written_twice_stores_once(env):
    with open(env.thread.outputs_path + "/dup.txt", "w") as f:
        f.write("v2")
    artifacts = [
        {"kind": "file", "path": "/mnt/user-data/outputs/dup.txt"},
        {"kind": "file", "path": "/mnt/user-data/outputs/dup.txt"},
    ]
    session = _Session()
    store_run_artifacts(session, env.conv, env.ctx, _outcome(artifacts))
    rows = [r for r in session.added if isinstance(r, StoredFile)]
    assert len(rows) == 1
    assert artifacts[0]["file_id"] == artifacts[1]["file_id"] == rows[0].id


def test_event_run_without_session_promotes_artifact(env):
    with open(env.thread.outputs_path + "/worker.txt", "w") as f:
        f.write("done")
    artifacts = [{"kind": "file", "path": "/mnt/user-data/outputs/worker.txt"}]
    session = _Session()

    store_run_artifacts(session, None, env.ctx, _outcome(artifacts))

    row = next(r for r in session.added if isinstance(r, StoredFile))
    assert row.event_id == "evt-artifacts"
    assert row.origin_session_id is None
    assert row.conversation_id is None
