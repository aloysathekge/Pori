"""Background (async) delegation — backs ``delegate_task(background=true)``.

A parent dispatches a child that runs on its own daemon thread and returns a handle
immediately, so the parent (and user) keep working. Completed children are queued;
the CLI drains them between turns and surfaces the results as a fresh turn — the
Hermes async-delegation model, on Pori's machinery.

Each child runs via ``asyncio.run`` on a fresh loop in its own thread — the same
path the synchronous delegate runner uses, minus the join.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, List, Optional


@dataclass
class BackgroundResult:
    handle: str
    goal: str
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None


class BackgroundDelegationRegistry:
    """Fire-and-forget child runs plus a drainable completion queue."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counter = itertools.count(1)
        self._active: Dict[str, str] = {}
        self._completed: List[BackgroundResult] = []

    def dispatch(
        self, goal: str, coro_factory: Callable[[], Coroutine[Any, Any, str]]
    ) -> str:
        """Start ``coro_factory()`` on its own daemon thread; return a handle now."""
        with self._lock:
            handle = f"bg-{next(self._counter)}"
            self._active[handle] = goal

        def target() -> None:
            try:
                result: str = asyncio.run(coro_factory())
                self._finish(BackgroundResult(handle, goal, True, result=result))
            except Exception as exc:  # captured, surfaced on drain
                self._finish(BackgroundResult(handle, goal, False, error=str(exc)))

        threading.Thread(target=target, daemon=True).start()
        return handle

    def _finish(self, res: BackgroundResult) -> None:
        with self._lock:
            self._active.pop(res.handle, None)
            self._completed.append(res)

    def drain_completed(self) -> List[BackgroundResult]:
        """Return + clear the finished children (the CLI calls this between turns)."""
        with self._lock:
            done, self._completed = self._completed, []
            return done

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def list_active(self) -> List[str]:
        with self._lock:
            return [f"{handle}: {goal}" for handle, goal in self._active.items()]
