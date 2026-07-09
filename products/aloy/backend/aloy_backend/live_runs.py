"""In-process registry of LIVE (in-flight) streamed runs, for re-attach.

Every SSE frame a run produces is published here as well as to the original
client. If the user navigates away and comes back, the live endpoint replays
the buffered frames and continues streaming — ChatGPT-style resume. A pending
clarification survives too (its request frame is in the replay, and the
clarify bridge stays registered until the run ends).

NOTE: in-process (single-worker constraint, same as CLARIFY_BRIDGES) — move to
a shared store (Redis pub/sub) to scale out. Entries retire a couple of
minutes after the run finishes; a late visitor just loads the persisted
conversation instead.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Dict, List, Optional, Set, Tuple

_RETIRE_AFTER_SECONDS = 120


class LiveRun:
    """One in-flight run's frame buffer + live subscribers (serving-loop only)."""

    def __init__(self, run_id: str, cancel: Optional[Callable[[], None]] = None):
        self.run_id = run_id
        self.history: List[str] = []
        self.queues: Set[asyncio.Queue] = set()
        self.done = False
        self._cancel = cancel

    def request_cancel(self) -> bool:
        """Ask the run to stop (cooperative — it halts at the next step
        boundary and the stream finishes normally). False if not stoppable."""
        if self.done or self._cancel is None:
            return False
        self._cancel()
        return True

    def publish(self, frame: str) -> None:
        self.history.append(frame)
        for q in self.queues:
            q.put_nowait(frame)

    def finish(self) -> None:
        self.done = True
        for q in self.queues:
            q.put_nowait(None)  # end-of-stream sentinel

    def subscribe(self) -> Tuple[List[str], asyncio.Queue]:
        """Atomic snapshot + live queue (single-threaded on the serving loop)."""
        q: asyncio.Queue = asyncio.Queue()
        replay = list(self.history)
        if self.done:
            q.put_nowait(None)
        else:
            self.queues.add(q)
        return replay, q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.queues.discard(q)


_LIVE: Dict[str, LiveRun] = {}  # conversation_id -> the conversation's live run


def register(
    conversation_id: str, run_id: str, cancel: Optional[Callable[[], None]] = None
) -> LiveRun:
    live = LiveRun(run_id, cancel=cancel)
    _LIVE[conversation_id] = live
    return live


def get(conversation_id: str) -> Optional[LiveRun]:
    return _LIVE.get(conversation_id)


def retire_later(conversation_id: str, live: LiveRun) -> None:
    """Drop the entry after a grace window (lets a just-returned client replay
    the finished stream instead of racing the DB persist)."""

    def _retire() -> None:
        if _LIVE.get(conversation_id) is live:
            del _LIVE[conversation_id]

    asyncio.get_running_loop().call_later(_RETIRE_AFTER_SECONDS, _retire)
