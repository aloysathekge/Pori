"""Trace storage backends."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from .trace import Trace


class TraceStore(ABC):
    """Abstract interface for trace persistence."""

    @abstractmethod
    async def save_trace(self, trace: Trace) -> None: ...

    @abstractmethod
    async def get_trace(self, trace_id: str) -> Optional[Trace]: ...

    @abstractmethod
    async def list_traces(
        self,
        agent_id: Optional[str] = None,
        limit: int = 20,
        page: int = 1,
    ) -> Tuple[List[Trace], int]: ...


class InMemoryTraceStore(TraceStore):
    """Simple in-memory store for development."""

    def __init__(self):
        self._traces: Dict[str, Trace] = {}

    async def save_trace(self, trace: Trace) -> None:
        self._traces[trace.trace_id] = trace

    async def get_trace(self, trace_id: str) -> Optional[Trace]:
        return self._traces.get(trace_id)

    async def list_traces(
        self,
        agent_id: Optional[str] = None,
        limit: int = 20,
        page: int = 1,
    ) -> Tuple[List[Trace], int]:
        traces = list(self._traces.values())
        if agent_id:
            traces = [t for t in traces if t.agent_id == agent_id]
        total = len(traces)
        start = (page - 1) * limit
        return traces[start : start + limit], total
