"""Telemetry exporters for traces and metrics."""

from abc import ABC, abstractmethod

from .trace import Trace


class TelemetryExporter(ABC):
    """Export traces/metrics to external observability platforms."""

    @abstractmethod
    async def export_trace(self, trace: Trace) -> None: ...

    @abstractmethod
    async def export_metrics(self, metrics) -> None: ...


class ConsoleTelemetryExporter(TelemetryExporter):
    """Print traces/metrics to console (for development)."""

    async def export_trace(self, trace: Trace) -> None:
        print(
            f"[TRACE] {trace.name} | {trace.duration_seconds:.2f}s | "
            f"{trace.total_spans} spans | status={trace.status.value}"
        )

    async def export_metrics(self, metrics) -> None:
        if hasattr(metrics, "summary"):
            print(f"[METRICS] {metrics.summary()}")
        else:
            print(f"[METRICS] {metrics}")
