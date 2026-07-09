"""Coalesce a run's live PoriEvent stream into a compact, replayable log.

Streamed token deltas (``text_delta`` / ``thinking_delta``) arrive one token at
a time — persisting each is row/JSON explosion. The collector buffers consecutive
same-type deltas and flushes them into a single ``text`` / ``thinking`` block the
moment a different event arrives, so ordering (a burst of reasoning → a tool call
→ its result) is preserved at the granularity a replay viewer actually shows.
Structural events (steps, tool calls, retries, run end) are kept verbatim.

The final answer text is already stored on the Message; this log exists for the
"peek at how the agent got there" replay, so it is bounded and best-effort.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pori import PoriEvent

_TEXT = "text_delta"
_THINKING = "thinking_delta"
_COALESCE = {_TEXT: "text", _THINKING: "thinking"}

# Bounds so a runaway run can't produce an unbounded log.
MAX_EVENTS = 4000
MAX_BLOCK_CHARS = 20000


class EventLogCollector:
    """Accumulates a coalesced, ordered event list for one run."""

    def __init__(self) -> None:
        self._events: List[Dict[str, Any]] = []
        self._buf: Optional[str] = None  # the coalesce key ("text"/"thinking")
        self._buf_text: List[str] = []
        self._buf_step: int = 0
        self._truncated = False

    def record(self, event: PoriEvent) -> None:
        if self._truncated or len(self._events) >= MAX_EVENTS:
            self._truncated = True
            return
        coalesce_key = _COALESCE.get(event.type)
        if coalesce_key is not None:
            if self._buf is not None and self._buf != coalesce_key:
                self._flush()
            self._buf = coalesce_key
            self._buf_step = event.step
            piece = str((event.payload or {}).get("text", ""))
            if piece:
                self._buf_text.append(piece)
            return
        # Structural event: flush any pending delta block, then keep verbatim.
        self._flush()
        self._events.append(
            {"type": event.type, "payload": event.payload or {}, "step": event.step}
        )

    def _flush(self) -> None:
        if self._buf is None:
            return
        text = "".join(self._buf_text)[:MAX_BLOCK_CHARS]
        if text:
            self._events.append(
                {"type": self._buf, "payload": {"text": text}, "step": self._buf_step}
            )
        self._buf = None
        self._buf_text = []

    def finalize(self) -> List[Dict[str, Any]]:
        self._flush()
        if self._truncated:
            self._events.append(
                {
                    "type": "truncated",
                    "payload": {"reason": f"event log capped at {MAX_EVENTS}"},
                    "step": 0,
                }
            )
        return self._events


__all__ = ["EventLogCollector"]
