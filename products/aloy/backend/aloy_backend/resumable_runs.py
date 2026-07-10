"""In-process cache of STOPPED runs' warm state, for true continue/resume.

When the user stops a run mid-generation, the run's ``AgentMemory`` (with all
its tool work) and its kernel task checkpoint stay warm here for a grace
window. A follow-up "continue" claims the entry and resumes the SAME kernel
task from its per-step checkpoint (``execute_task(resume_task_id=…)``), so a
run stopped at step 6 of 15 picks up at step 6 instead of redoing the work.
A cold cache is fine: the continuation turn alone still works, because the
stopped run's partial text is persisted into conversation history.

NOTE: in-process (single-worker constraint, same as live_runs and
CLARIFY_BRIDGES) — move to a shared store to scale out.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

_EXPIRE_AFTER_SECONDS = 30 * 60


@dataclass
class ResumableRun:
    run_id: str  # the stopped run — message metadata carries it to the UI
    task: str  # the original task the agent was executing
    task_id: str  # kernel task id, for execute_task(resume_task_id=…)
    memory: Any  # the run's live AgentMemory (tool work intact)


_STOPPED: Dict[str, ResumableRun] = {}  # conversation_id -> latest stopped run


def register(conversation_id: str, entry: ResumableRun) -> None:
    _STOPPED[conversation_id] = entry

    def _expire() -> None:
        if _STOPPED.get(conversation_id) is entry:
            del _STOPPED[conversation_id]

    asyncio.get_running_loop().call_later(_EXPIRE_AFTER_SECONDS, _expire)


def claim(conversation_id: str, run_id: str) -> Optional[ResumableRun]:
    """One-shot: pop and return the entry iff it matches the requested run."""
    entry = _STOPPED.get(conversation_id)
    if entry is None or entry.run_id != run_id:
        return None
    del _STOPPED[conversation_id]
    return entry


def discard(conversation_id: str) -> None:
    """The conversation moved on (a new normal turn): resuming the old stopped
    run later would fork history, so drop its warm state."""
    _STOPPED.pop(conversation_id, None)
