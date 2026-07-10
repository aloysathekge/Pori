"""The warm-resume cache: a stopped run's state is claimable exactly once,
only by its own run_id, and a normal turn invalidates it."""

import pytest

from aloy_backend import resumable_runs

pytestmark = pytest.mark.asyncio


def _entry(run_id: str) -> resumable_runs.ResumableRun:
    return resumable_runs.ResumableRun(
        run_id=run_id, task="original task", task_id="k1", memory=object()
    )


async def test_claim_is_one_shot_and_run_scoped():
    entry = _entry("run-1")
    resumable_runs.register("conv-1", entry)

    assert resumable_runs.claim("conv-1", "someone-else") is None
    assert resumable_runs.claim("conv-1", "run-1") is entry
    # One-shot: a second claim finds nothing.
    assert resumable_runs.claim("conv-1", "run-1") is None


async def test_discard_invalidates_warm_state():
    resumable_runs.register("conv-2", _entry("run-2"))
    resumable_runs.discard("conv-2")
    assert resumable_runs.claim("conv-2", "run-2") is None


async def test_new_stop_replaces_older_entry():
    old, new = _entry("run-old"), _entry("run-new")
    resumable_runs.register("conv-3", old)
    resumable_runs.register("conv-3", new)
    assert resumable_runs.claim("conv-3", "run-old") is None
    assert resumable_runs.claim("conv-3", "run-new") is new
