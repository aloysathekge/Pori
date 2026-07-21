"""Normalized agent event stream.

A single event shape that every provider adapter emits and every renderer
(CLI, Cloud SSE, JSONL) consumes, so live UX and the replay/audit trail come
from the same source. See
``.agent/reference-studies/streaming-event-architecture.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict

# --- event types ------------------------------------------------------------
RUN_START = "run_start"
RUN_END = "run_end"
STEP_START = "step_start"
STEP_END = "step_end"
TEXT_DELTA = "text_delta"  # visible answer prose, streamed
THINKING_DELTA = "thinking_delta"  # reasoning prose, streamed (may be dimmed)
TOOL_CALL_START = "tool_call_start"  # the instant the tool name is known
TOOL_CALL_END = "tool_call_end"  # after execution: success + result
LLM_RETRY = "llm_retry"  # API retrying/rate-limited (not "still thinking")
ACTIVITY_CHANGED = "activity_changed"  # model-authored intent for current work
PLAN_CHANGED = "plan_changed"  # host-observed normalized plan snapshot


@dataclass
class PoriEvent:
    """One normalized agent event.

    Providers emit ``type`` + ``payload``; the agent enriches with ``step`` when
    it forwards to consumers. ``payload`` is event-specific, e.g.
    ``{"text": "..."}`` for TEXT_DELTA or ``{"name": "answer"}`` for
    TOOL_CALL_START.
    """

    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    step: int = 0


class JsonlEventSink:
    """Append PoriEvents as JSON lines for replay/audit.

    A renderer-agnostic consumer: compose it alongside a live renderer so the
    same event stream that drives the UI also produces a diff-able trail. Opens
    the file per event (simple + crash-safe); runs are bounded so this is cheap.
    """

    def __init__(self, path: str):
        self.path = path

    def __call__(self, event: "PoriEvent") -> None:
        try:
            line = json.dumps(
                {"type": event.type, "payload": event.payload, "step": event.step}
            )
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            # Logging must never break a run.
            pass
