"""Span-based tracing for agent execution."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class SpanType(str, Enum):
    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    EVAL = "eval"
    GUARDRAIL = "guardrail"


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


@dataclass
class Span:
    """A single unit of work in a trace."""

    span_id: str = field(default_factory=lambda: str(uuid4())[:12])
    parent_span_id: Optional[str] = None
    trace_id: str = ""
    name: str = ""
    span_type: SpanType = SpanType.AGENT
    status: SpanStatus = SpanStatus.OK

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    attributes: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    children: List["Span"] = field(default_factory=list)

    def finish(self, status: SpanStatus = SpanStatus.OK, error: Optional[str] = None):
        self.end_time = datetime.now()
        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.status = status
        self.error = error


@dataclass
class Trace:
    """A complete execution trace (tree of spans)."""

    trace_id: str = field(default_factory=lambda: str(uuid4())[:12])
    name: str = ""
    run_id: str = ""
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    team_id: Optional[str] = None

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    status: SpanStatus = SpanStatus.OK

    root_spans: List[Span] = field(default_factory=list)
    _all_spans: List[Span] = field(default_factory=list, repr=False)

    input: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict] = None,
    ) -> Span:
        """Create and start a new span."""
        span = Span(
            trace_id=self.trace_id,
            parent_span_id=parent_span_id,
            name=name,
            span_type=span_type,
            start_time=datetime.now(),
            attributes=attributes or {},
        )

        if parent_span_id:
            parent = next(
                (s for s in self._all_spans if s.span_id == parent_span_id), None
            )
            if parent:
                parent.children.append(span)
        else:
            self.root_spans.append(span)

        self._all_spans.append(span)
        return span

    def finish(self):
        self.end_time = datetime.now()
        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        if any(s.status == SpanStatus.ERROR for s in self._all_spans):
            self.status = SpanStatus.ERROR

    @property
    def total_spans(self) -> int:
        return len(self._all_spans)

    @property
    def error_count(self) -> int:
        return sum(1 for s in self._all_spans if s.status == SpanStatus.ERROR)

    def to_dict(self) -> Dict:
        """Serialize for storage/API response."""

        def span_to_dict(span: Span) -> Dict:
            return {
                "span_id": span.span_id,
                "parent_span_id": span.parent_span_id,
                "name": span.name,
                "type": span.span_type.value,
                "status": span.status.value,
                "duration": f"{span.duration_seconds:.3f}s",
                "attributes": span.attributes,
                "error": span.error,
                "children": [span_to_dict(c) for c in span.children],
            }

        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "duration": f"{self.duration_seconds:.3f}s",
            "total_spans": self.total_spans,
            "error_count": self.error_count,
            "input": self.input,
            "output": self.output,
            "tree": [span_to_dict(s) for s in self.root_spans],
        }
