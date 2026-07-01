"""Streaming reasoning-tag scrubber for 'tagged' reasoning models.

Some open-weight models emit their reasoning inline as ``<think>...</think>``
inside the same text stream (DeepSeek-R1 style). This splits those spans out of
a *streamed* text — surviving chunk boundaries — so a renderer can show the
reasoning as a separate (e.g. dimmed) block and keep it out of the final answer.
See ``.agent/reference-studies/streaming-event-architecture.md`` §4.
"""

from __future__ import annotations

from typing import List, Tuple

_OPEN = "<think>"
_CLOSE = "</think>"
_MAX_TAG = max(len(_OPEN), len(_CLOSE))

# (kind, text) where kind is "thinking" or "text".
Segment = Tuple[str, str]


class StreamingThinkScrubber:
    """Split ``<think>...</think>`` out of a streamed text, statefully.

    ``feed(chunk)`` returns the segments decided so far; a trailing partial tag
    is held back until the next chunk. ``flush()`` emits whatever remains.
    """

    def __init__(self) -> None:
        self._in_think = False
        self._buf = ""

    def _kind(self) -> str:
        return "thinking" if self._in_think else "text"

    def feed(self, chunk: str) -> List[Segment]:
        self._buf += chunk
        out: List[Segment] = []
        # Consume every complete tag currently in the buffer.
        while True:
            tag = _CLOSE if self._in_think else _OPEN
            idx = self._buf.find(tag)
            if idx == -1:
                break
            before = self._buf[:idx]
            if before:
                out.append((self._kind(), before))
            self._buf = self._buf[idx + len(tag) :]
            self._in_think = not self._in_think
        # Emit everything except a tail that could still be the start of a tag.
        hold = _MAX_TAG - 1
        if len(self._buf) > hold:
            cut = len(self._buf) - hold
            emit, self._buf = self._buf[:cut], self._buf[cut:]
            if emit:
                out.append((self._kind(), emit))
        return out

    def flush(self) -> List[Segment]:
        out: List[Segment] = []
        if self._buf:
            out.append((self._kind(), self._buf))
            self._buf = ""
        return out
