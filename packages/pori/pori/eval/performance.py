"""Performance evaluation — runtime and memory benchmarking."""

import asyncio
import gc
import statistics
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional

from .base import BaseEval, EvalResult


@dataclass
class PerformanceResult(EvalResult):
    """Result of a performance evaluation."""

    run_times: List[float] = field(default_factory=list)
    memory_usages: List[float] = field(default_factory=list)
    avg_run_time: float = 0.0
    min_run_time: float = 0.0
    max_run_time: float = 0.0
    p95_run_time: float = 0.0
    avg_memory: float = 0.0


class PerformanceEval(BaseEval):
    """Benchmark an agent's runtime performance and memory usage.

    Runs the provided function multiple times and collects statistics.

    Usage:
        eval = PerformanceEval(
            func=lambda: agent.run(),
            num_iterations=10,
            warmup_runs=2,
        )
        result = await eval.run()
        assert result.avg_run_time < 5.0
    """

    def __init__(
        self,
        func: Callable,
        num_iterations: int = 10,
        warmup_runs: int = 2,
        measure_memory: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.func = func
        self.num_iterations = num_iterations
        self.warmup_runs = warmup_runs
        self.measure_memory = measure_memory

    async def run(self, **kwargs) -> PerformanceResult:
        # Warmup
        for _ in range(self.warmup_runs):
            result = self.func()
            if asyncio.iscoroutine(result):
                await result

        run_times = []
        memory_usages = []

        for _ in range(self.num_iterations):
            start = datetime.now()

            if self.measure_memory:
                gc.collect()
                tracemalloc.start()

            result = self.func()
            if asyncio.iscoroutine(result):
                await result

            duration = (datetime.now() - start).total_seconds()
            run_times.append(duration)

            if self.measure_memory:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory_usages.append(peak / 1024 / 1024)  # MiB

        avg_time = statistics.mean(run_times)
        sorted_times = sorted(run_times)
        p95_idx = int(len(sorted_times) * 0.95)
        p95 = sorted_times[min(p95_idx, len(sorted_times) - 1)]

        return PerformanceResult(
            eval_id=self.eval_id,
            eval_type="performance",
            passed=True,  # Performance evals don't inherently pass/fail
            data={
                "num_iterations": self.num_iterations,
                "warmup_runs": self.warmup_runs,
            },
            run_times=run_times,
            memory_usages=memory_usages,
            avg_run_time=avg_time,
            min_run_time=min(run_times),
            max_run_time=max(run_times),
            p95_run_time=p95,
            avg_memory=statistics.mean(memory_usages) if memory_usages else 0.0,
        )
