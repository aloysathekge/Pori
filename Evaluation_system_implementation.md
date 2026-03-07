# Enterprise Systems Implementation Plan for Pori

## Executive Summary

This plan describes how to add **observability**, **evaluations**, and **guardrails** to Pori, based on patterns learned from agno's production-grade implementations. These three systems are the pillars enterprises require before adopting an agent framework:

1. **Observability** — Know what your agents are doing (metrics, traces, telemetry)
2. **Evaluations** — Prove your agents work correctly (accuracy, reliability, performance)
3. **Guardrails** — Ensure your agents behave safely (input/output validation, policy enforcement)

The work is broken into **4 phases**, each independently shippable. The design leverages Pori's existing primitives (`Agent`, `AgentMemory`, `ToolRegistry`, `Evaluator`, `ActionResult`) rather than building from scratch.

---

## How Agno Does It (Patterns to Learn From)

### Agno's Observability Stack (3 Layers)

| Layer | What it tracks | Where it lives |
|-------|---------------|----------------|
| **Metrics** | Token counts, cost, latency, TTFT per LLM call, per run, per session | `agno/metrics.py` — `BaseMetrics`, `MessageMetrics`, `RunMetrics`, `SessionMetrics` |
| **Telemetry** | Agent/team configuration snapshots shipped to cloud | `agno/agent/_telemetry.py`, `agno/team/_telemetry.py` |
| **Traces** | Hierarchical span trees (Agent > LLM > Tool) with timing, status, I/O | `agno/os/routers/traces/traces.py` — OpenTelemetry-style span hierarchy |

Key design: Metrics classes implement `__add__` so they're additive across calls, runs, and sessions. Every LLM invocation automatically populates a `MessageMetrics` object that rolls up into `RunMetrics`.

### Agno's Evaluation Framework (4 Types)

| Eval Type | What it tests | LLM needed? | Key class |
|-----------|--------------|-------------|-----------|
| **Accuracy** | "Did the agent get the right answer?" | Yes (evaluator agent scores 1-10) | `AccuracyEval` |
| **Reliability** | "Did the agent call the right tools?" | No (deterministic check) | `ReliabilityEval` |
| **Performance** | "How fast/memory-efficient is the agent?" | No (timer + tracemalloc) | `PerformanceEval` |
| **Agent-as-Judge** | "Does output meet custom criteria?" | Yes (judge agent with configurable scoring) | `AgentAsJudgeEval` |

All evals share a common pattern: run agent, collect result, score it, store in DB, optionally send telemetry.

### Agno's Guardrails (Unified with Evals)

Agno doesn't have a separate "guardrails" module. Instead, the `BaseEval` abstract class provides:

```python
class BaseEval(ABC):
    def pre_check(self, run_input) -> None:   # Input guardrail (before agent runs)
    def post_check(self, run_output) -> None:  # Output guardrail (after agent runs)
```

The `AgentAsJudgeEval` implements these as runtime safety checks. When attached to an agent, it validates inputs before processing and outputs before returning to the user. Same evaluation logic works both as a runtime guardrail and as an offline quality test.

---

## Phase 1: Run Metrics & Cost Tracking

**Goal**: Every `Agent.run()` call automatically collects and returns structured metrics.

**Effort**: ~2-3 days

### 1.1 Create `pori/metrics.py` — Core Metrics Data Model

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class TokenUsage:
    """Token consumption for a single LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM invocation."""
    model_id: str = ""
    model_provider: str = ""
    tokens: TokenUsage = field(default_factory=TokenUsage)
    cost: Optional[float] = None          # USD cost for this call
    duration_seconds: float = 0.0          # Wall-clock time
    time_to_first_token: Optional[float] = None  # For streaming
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToolCallMetrics:
    """Metrics for a single tool execution."""
    tool_name: str = ""
    duration_seconds: float = 0.0
    success: bool = True
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StepMetrics:
    """Metrics for a single agent step (may contain multiple LLM/tool calls)."""
    step_number: int = 0
    llm_calls: List[LLMCallMetrics] = field(default_factory=list)
    tool_calls: List[ToolCallMetrics] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def total_tokens(self) -> TokenUsage:
        result = TokenUsage()
        for call in self.llm_calls:
            result = result + call.tokens
        return result

    @property
    def total_cost(self) -> float:
        return sum(c.cost or 0.0 for c in self.llm_calls)


@dataclass
class RunMetrics:
    """Aggregate metrics for an entire Agent.run() execution."""
    run_id: str = ""
    agent_id: str = ""
    agent_name: Optional[str] = None
    model_id: str = ""
    model_provider: str = ""

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Per-step breakdown
    steps: List[StepMetrics] = field(default_factory=list)

    # Aggregates (computed)
    total_steps: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0

    @property
    def total_tokens(self) -> TokenUsage:
        result = TokenUsage()
        for step in self.steps:
            result = result + step.total_tokens
        return result

    @property
    def total_cost(self) -> float:
        return sum(step.total_cost for step in self.steps)

    def finalize(self) -> None:
        """Compute aggregate fields after run completes."""
        self.total_steps = len(self.steps)
        self.total_llm_calls = sum(len(s.llm_calls) for s in self.steps)
        self.total_tool_calls = sum(len(s.tool_calls) for s in self.steps)
        if self.start_time and self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()

    def summary(self) -> Dict:
        """Return a human-readable summary dict."""
        tokens = self.total_tokens
        return {
            "run_id": self.run_id,
            "agent": self.agent_name or self.agent_id,
            "model": f"{self.model_provider}/{self.model_id}",
            "duration": f"{self.duration_seconds:.2f}s",
            "steps": self.total_steps,
            "llm_calls": self.total_llm_calls,
            "tool_calls": self.total_tool_calls,
            "tokens": {
                "input": tokens.input_tokens,
                "output": tokens.output_tokens,
                "total": tokens.total_tokens,
            },
            "cost_usd": f"${self.total_cost:.4f}",
        }
```

**Why this design?**
- `TokenUsage` implements `__add__` so metrics accumulate naturally (same pattern as agno's `BaseMetrics`)
- Hierarchical: `RunMetrics` > `StepMetrics` > `LLMCallMetrics` / `ToolCallMetrics`
- The `summary()` method gives you a one-liner for logging/dashboards
- Cost tracking is optional (set per-model pricing or leave as None)

### 1.2 Integrate Metrics into `Agent.step()` and `Agent.run()`

**In `Agent.step()`** — wrap LLM calls and tool executions with timing:

```python
async def step(self) -> None:
    step_metrics = StepMetrics(step_number=self.state.n_steps + 1)
    step_start = datetime.now()

    # ... existing step logic ...

    # When calling LLM (get_next_action):
    llm_start = datetime.now()
    model_output = await self.get_next_action()
    llm_metrics = LLMCallMetrics(
        model_id=self.llm.model_id,        # Need to expose from BaseChatModel
        model_provider=self.llm.provider,   # Need to expose from BaseChatModel
        duration_seconds=(datetime.now() - llm_start).total_seconds(),
        tokens=self._extract_token_usage(response),  # Parse from LLM response
    )
    step_metrics.llm_calls.append(llm_metrics)

    # When executing tools:
    for action in model_output.action:
        tool_start = datetime.now()
        result = await self._execute_single_action(action)
        tool_metrics = ToolCallMetrics(
            tool_name=action_name,
            duration_seconds=(datetime.now() - tool_start).total_seconds(),
            success=result.success,
        )
        step_metrics.tool_calls.append(tool_metrics)

    step_metrics.duration_seconds = (datetime.now() - step_start).total_seconds()
    self._run_metrics.steps.append(step_metrics)
```

**In `Agent.run()`** — initialize and finalize `RunMetrics`:

```python
async def run(self) -> Dict[str, Any]:
    self._run_metrics = RunMetrics(
        run_id=self.task_id,
        agent_id=self.agent_id,
        agent_name=self.name,
        model_id=self.llm.model_id,
        model_provider=self.llm.provider,
        start_time=datetime.now(),
    )

    # ... existing run loop ...

    self._run_metrics.end_time = datetime.now()
    self._run_metrics.finalize()

    return {
        "completed": is_complete,
        "steps_taken": self.state.n_steps,
        "metrics": self._run_metrics,           # NEW
        "metrics_summary": self._run_metrics.summary(),  # NEW
    }
```

### 1.3 Expose Token Usage from `BaseChatModel`

Pori's `BaseChatModel` currently doesn't return token usage from LLM responses. Add a standard way to extract it:

```python
# In pori/llm/base.py or wherever BaseChatModel is defined
class LLMResponse:
    """Standardized response wrapper."""
    content: Any
    raw: Any  # Original provider response
    token_usage: Optional[TokenUsage] = None  # Extracted token counts

# Or simpler: add a method to BaseChatModel
def extract_token_usage(self, raw_response: Any) -> TokenUsage:
    """Extract token usage from provider-specific response. Override per provider."""
    return TokenUsage()
```

For Anthropic: `response.usage.input_tokens`, `response.usage.output_tokens`
For OpenAI: `response.usage.prompt_tokens`, `response.usage.completion_tokens`

### 1.4 Cost Calculation (Optional but Enterprise-Valuable)

```python
# pori/metrics.py

# Simple pricing table (USD per 1M tokens)
MODEL_PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

def calculate_cost(model_id: str, tokens: TokenUsage) -> Optional[float]:
    """Calculate USD cost for a set of token usage."""
    pricing = MODEL_PRICING.get(model_id)
    if not pricing:
        return None
    input_cost = (tokens.input_tokens / 1_000_000) * pricing["input"]
    output_cost = (tokens.output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)
```

### Files to create/modify:
- **Create**: `pori/metrics.py`
- **Modify**: `pori/agent.py` — integrate `RunMetrics` into step/run loop
- **Modify**: `pori/llm/base.py` (or equivalent) — expose token usage extraction

---

## Phase 2: Evaluation Framework

**Goal**: Provide 4 evaluation types that test agent quality, matching agno's evaluation capabilities.

**Effort**: ~4-5 days

### 2.1 Create `pori/eval/base.py` — Base Evaluation Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import uuid4


@dataclass
class EvalResult:
    """Base result from any evaluation."""
    eval_id: str
    eval_type: str
    passed: bool
    data: Dict[str, Any] = field(default_factory=dict)

    def assert_passed(self):
        assert self.passed, f"Evaluation {self.eval_type} FAILED: {self.data}"


class BaseEval(ABC):
    """Abstract base for all evaluation types.

    Also serves as the guardrail interface:
    - pre_check(): input guardrail (before agent runs)
    - post_check(): output guardrail (after agent runs)
    """

    def __init__(self, name: Optional[str] = None):
        self.eval_id = str(uuid4())
        self.name = name

    # --- Standalone evaluation ---

    @abstractmethod
    async def run(self, **kwargs) -> EvalResult:
        """Run the evaluation and return a result."""
        ...

    # --- Guardrail hooks (for runtime use) ---

    async def pre_check(self, input_text: str) -> None:
        """Validate input before the agent processes it.
        Raise an exception to block the run.
        Override in subclasses to implement input guardrails.
        """
        pass

    async def post_check(self, input_text: str, output_text: str) -> None:
        """Validate output before returning to the user.
        Raise an exception to block the response.
        Override in subclasses to implement output guardrails.
        """
        pass
```

**Key insight from agno**: The same `BaseEval` class serves both as a standalone evaluation AND as a runtime guardrail. `pre_check`/`post_check` are the guardrail hooks. This means you don't need a separate guardrails system.

### 2.2 Create `pori/eval/reliability.py` — Tool Call Verification

**Start here** because it's the simplest (no LLM needed, purely deterministic).

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pori.eval.base import BaseEval, EvalResult


@dataclass
class ReliabilityResult(EvalResult):
    """Result of a reliability evaluation."""
    passed_tool_calls: List[str] = field(default_factory=list)
    failed_tool_calls: List[str] = field(default_factory=list)


class ReliabilityEval(BaseEval):
    """
    Test whether the agent calls the expected tools for a given input.
    No LLM judge needed - purely deterministic comparison.

    Usage:
        eval = ReliabilityEval(
            agent=my_agent,
            input="multiply 3 by 7",
            expected_tool_calls=["multiply"],
        )
        result = await eval.run()
        result.assert_passed()
    """

    def __init__(
        self,
        agent: "Agent",
        input: str,
        expected_tool_calls: List[str],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.agent = agent
        self.input = input
        self.expected_tool_calls = expected_tool_calls

    async def run(self, **kwargs) -> ReliabilityResult:
        # Run the agent
        run_result = await self.agent.run()  # Agent was initialized with self.input as task

        # Extract actual tool calls from memory
        actual_tools = [
            tc.tool_name for tc in self.agent.memory.tool_call_history
        ]

        # Compare
        passed = []
        failed = []
        for expected in self.expected_tool_calls:
            if expected in actual_tools:
                passed.append(expected)
            else:
                failed.append(expected)

        return ReliabilityResult(
            eval_id=self.eval_id,
            eval_type="reliability",
            passed=len(failed) == 0,
            data={
                "expected": self.expected_tool_calls,
                "actual": actual_tools,
            },
            passed_tool_calls=passed,
            failed_tool_calls=failed,
        )
```

### 2.3 Create `pori/eval/accuracy.py` — LLM-Judged Answer Quality

This uses a **separate evaluator agent** to score the output against an expected answer — the same pattern agno uses.

```python
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from pori.eval.base import BaseEval, EvalResult
from pori.llm import BaseChatModel


class AccuracyScore(BaseModel):
    """Structured output from the evaluator agent."""
    score: int = Field(..., ge=1, le=10, description="Accuracy score 1-10")
    reason: str = Field(..., description="Reasoning for the score")


@dataclass
class AccuracyResult(EvalResult):
    """Result of an accuracy evaluation."""
    score: float = 0.0
    reason: str = ""
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    avg_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0


class AccuracyEval(BaseEval):
    """
    Test whether the agent's output matches an expected answer.
    Uses a separate LLM as judge to score accuracy 1-10.

    Usage:
        eval = AccuracyEval(
            agent=my_agent,
            input="What is 2+2?",
            expected_output="4",
            evaluator_llm=my_evaluator_model,  # e.g., GPT-4o or Claude
        )
        result = await eval.run()
        assert result.avg_score >= 7
    """

    def __init__(
        self,
        agent: "Agent",
        input: str,
        expected_output: str,
        evaluator_llm: BaseChatModel,
        num_iterations: int = 1,
        additional_guidelines: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.agent = agent
        self.input = input
        self.expected_output = expected_output
        self.evaluator_llm = evaluator_llm
        self.num_iterations = num_iterations
        self.additional_guidelines = additional_guidelines or ""

    async def run(self, **kwargs) -> AccuracyResult:
        import statistics

        iterations = []

        for i in range(self.num_iterations):
            # 1. Run the agent
            run_result = await self.agent.run()
            agent_output = str(
                self.agent.memory.get_final_answer() or "No answer provided"
            )

            # 2. Ask evaluator to score
            eval_prompt = f"""Compare the agent's output to the expected output.

<agent_input>{self.input}</agent_input>
<expected_output>{self.expected_output}</expected_output>
<agent_output>{agent_output}</agent_output>

{self.additional_guidelines}

Score the accuracy from 1-10:
1-2: Completely incorrect
3-4: Major inaccuracies
5-6: Partially correct with significant issues
7-8: Mostly accurate with minor issues
9-10: Highly accurate, matches expected output closely

You must assume the expected_output is correct."""

            structured = self.evaluator_llm.with_structured_output(
                AccuracyScore, include_raw=True
            )
            response = await structured.ainvoke([
                {"role": "system", "content": "You are an expert evaluator. Score objectively."},
                {"role": "user", "content": eval_prompt},
            ])

            parsed = response.get("parsed")
            if parsed:
                iterations.append({
                    "iteration": i + 1,
                    "agent_output": agent_output,
                    "score": parsed.score,
                    "reason": parsed.reason,
                })

        # Compute stats
        scores = [it["score"] for it in iterations]
        avg = statistics.mean(scores) if scores else 0
        mn = min(scores) if scores else 0
        mx = max(scores) if scores else 0

        return AccuracyResult(
            eval_id=self.eval_id,
            eval_type="accuracy",
            passed=avg >= 7,  # Default threshold
            data={"iterations": iterations},
            score=avg,
            reason=iterations[-1]["reason"] if iterations else "",
            iterations=iterations,
            avg_score=avg,
            min_score=mn,
            max_score=mx,
        )
```

### 2.4 Create `pori/eval/performance.py` — Runtime & Memory Benchmarking

```python
import asyncio
import gc
import tracemalloc
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from pori.eval.base import BaseEval, EvalResult


@dataclass
class PerformanceResult(EvalResult):
    """Result of a performance evaluation."""
    run_times: List[float] = field(default_factory=list)
    memory_usages: List[float] = field(default_factory=list)  # MiB
    avg_run_time: float = 0.0
    min_run_time: float = 0.0
    max_run_time: float = 0.0
    p95_run_time: float = 0.0
    avg_memory: float = 0.0


class PerformanceEval(BaseEval):
    """
    Benchmark an agent's runtime performance and memory usage.
    Runs the agent function multiple times and collects statistics.

    Usage:
        eval = PerformanceEval(
            func=lambda: agent.run(),
            num_iterations=10,
            warmup_runs=2,
        )
        result = await eval.run()
        assert result.avg_run_time < 5.0  # Under 5 seconds
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
        import statistics
        from datetime import datetime

        # Warmup
        for _ in range(self.warmup_runs):
            result = self.func()
            if asyncio.iscoroutine(result):
                await result

        run_times = []
        memory_usages = []

        for i in range(self.num_iterations):
            # Measure time
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

        # Stats
        avg_time = statistics.mean(run_times)
        sorted_times = sorted(run_times)
        p95 = sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 1 else sorted_times[0]

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
```

### 2.5 Create `pori/eval/agent_judge.py` — Custom Criteria Evaluation

This is the most flexible eval type. You define criteria and a judge agent scores the output.

```python
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from pori.eval.base import BaseEval, EvalResult
from pori.llm import BaseChatModel


class BinaryJudgment(BaseModel):
    """Binary pass/fail judgment."""
    passed: bool = Field(..., description="Whether the output passes the criteria")
    reason: str = Field(..., description="Reasoning for the judgment")


class NumericJudgment(BaseModel):
    """Numeric 1-10 judgment."""
    score: int = Field(..., ge=1, le=10, description="Score from 1-10")
    reason: str = Field(..., description="Reasoning for the score")


@dataclass
class AgentJudgeResult(EvalResult):
    """Result of an agent-as-judge evaluation."""
    score: Optional[int] = None
    reason: str = ""
    pass_rate: float = 0.0
    evaluations: List[Dict] = field(default_factory=list)


class AgentJudgeEval(BaseEval):
    """
    Evaluate agent output against custom criteria using an LLM judge.

    Supports two scoring modes:
    - "binary": PASS / FAIL
    - "numeric": Score 1-10 with configurable threshold

    Also implements pre_check / post_check for use as a runtime guardrail.

    Usage as eval:
        eval = AgentJudgeEval(
            criteria="Response must be professional and cite sources",
            judge_llm=my_model,
            scoring="binary",
        )
        result = await eval.run(input="...", output="...")

    Usage as guardrail (attach to agent):
        agent.guardrails = [
            AgentJudgeEval(
                criteria="No PII in responses",
                judge_llm=my_model,
                scoring="binary",
            )
        ]
    """

    def __init__(
        self,
        criteria: str,
        judge_llm: BaseChatModel,
        scoring: Literal["binary", "numeric"] = "binary",
        threshold: int = 7,  # For numeric: score >= threshold = pass
        on_fail: Optional[Callable] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.criteria = criteria
        self.judge_llm = judge_llm
        self.scoring = scoring
        self.threshold = threshold
        self.on_fail = on_fail

    async def run(
        self,
        input: str = "",
        output: str = "",
        cases: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> AgentJudgeResult:
        """Evaluate one or multiple input/output pairs."""
        pairs = cases or [{"input": input, "output": output}]
        evaluations = []

        schema = NumericJudgment if self.scoring == "numeric" else BinaryJudgment
        structured = self.judge_llm.with_structured_output(schema, include_raw=True)

        for pair in pairs:
            prompt = f"""Evaluate the following output against the criteria.

<criteria>{self.criteria}</criteria>
<input>{pair['input']}</input>
<output>{pair['output']}</output>

Be objective and thorough."""

            response = await structured.ainvoke([
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": prompt},
            ])

            parsed = response.get("parsed")
            if parsed:
                if self.scoring == "numeric":
                    passed = parsed.score >= self.threshold
                    evaluations.append({
                        "input": pair["input"],
                        "output": pair["output"],
                        "score": parsed.score,
                        "reason": parsed.reason,
                        "passed": passed,
                    })
                else:
                    evaluations.append({
                        "input": pair["input"],
                        "output": pair["output"],
                        "passed": parsed.passed,
                        "reason": parsed.reason,
                    })

                # Trigger on_fail callback
                if not evaluations[-1]["passed"] and self.on_fail:
                    self.on_fail(evaluations[-1])

        pass_count = sum(1 for e in evaluations if e["passed"])
        pass_rate = pass_count / len(evaluations) if evaluations else 0.0

        return AgentJudgeResult(
            eval_id=self.eval_id,
            eval_type="agent_judge",
            passed=pass_rate == 1.0,
            data={"criteria": self.criteria, "scoring": self.scoring},
            score=evaluations[-1].get("score") if evaluations else None,
            reason=evaluations[-1].get("reason", "") if evaluations else "",
            pass_rate=pass_rate,
            evaluations=evaluations,
        )

    # --- Guardrail hooks ---

    async def pre_check(self, input_text: str) -> None:
        """Use as input guardrail: validate input against criteria."""
        result = await self.run(input=input_text, output="(pre-check: input only)")
        if not result.passed:
            raise ValueError(
                f"Input guardrail failed: {result.reason}"
            )

    async def post_check(self, input_text: str, output_text: str) -> None:
        """Use as output guardrail: validate output against criteria."""
        result = await self.run(input=input_text, output=output_text)
        if not result.passed:
            raise ValueError(
                f"Output guardrail failed: {result.reason}"
            )
```

### Files to create:
- `pori/eval/__init__.py`
- `pori/eval/base.py`
- `pori/eval/reliability.py`
- `pori/eval/accuracy.py`
- `pori/eval/performance.py`
- `pori/eval/agent_judge.py`

---

## Phase 3: Guardrails (Runtime Safety Hooks)

**Goal**: Allow attaching evaluations as runtime guardrails on `Agent.run()`.

**Effort**: ~1-2 days

### 3.1 Add guardrail support to `Agent`

```python
class Agent:
    def __init__(
        self,
        task: str,
        llm: BaseChatModel,
        # ... existing params ...
        # NEW: guardrails
        guardrails: Optional[List[BaseEval]] = None,
    ):
        self.guardrails = guardrails or []
```

### 3.2 Run guardrail pre-checks before processing

In `Agent.run()`, before the main loop:

```python
async def run(self) -> Dict[str, Any]:
    # Pre-check guardrails (input validation)
    for guardrail in self.guardrails:
        try:
            await guardrail.pre_check(self.task)
        except ValueError as e:
            logger.warning(f"Input guardrail blocked: {e}")
            return {
                "completed": False,
                "blocked_by": "input_guardrail",
                "reason": str(e),
                "steps_taken": 0,
            }

    # ... existing run loop ...
```

### 3.3 Run guardrail post-checks before returning

After the run loop completes, before returning the result:

```python
    # Post-check guardrails (output validation)
    final_answer = self.memory.get_final_answer()
    if final_answer:
        output_text = str(final_answer)
        for guardrail in self.guardrails:
            try:
                await guardrail.post_check(self.task, output_text)
            except ValueError as e:
                logger.warning(f"Output guardrail blocked: {e}")
                return {
                    "completed": False,
                    "blocked_by": "output_guardrail",
                    "reason": str(e),
                    "steps_taken": self.state.n_steps,
                }

    return {
        "completed": is_complete,
        "steps_taken": self.state.n_steps,
        "metrics": self._run_metrics,
    }
```

### 3.4 Built-in Guardrails (Convenience)

Provide some common guardrails out of the box:

```python
# pori/eval/guardrails.py

class ContentPolicyGuardrail(AgentJudgeEval):
    """Block responses containing harmful content."""
    def __init__(self, judge_llm: BaseChatModel):
        super().__init__(
            criteria=(
                "The output must NOT contain: hate speech, explicit violence, "
                "personally identifiable information (PII), or instructions for "
                "illegal activities."
            ),
            judge_llm=judge_llm,
            scoring="binary",
            name="content_policy",
        )


class FactualityGuardrail(AgentJudgeEval):
    """Flag responses that make unverifiable claims."""
    def __init__(self, judge_llm: BaseChatModel):
        super().__init__(
            criteria=(
                "The output should only contain factual, verifiable claims. "
                "Flag responses that present speculation as fact or make "
                "unsubstantiated claims."
            ),
            judge_llm=judge_llm,
            scoring="binary",
            name="factuality",
        )


class TopicGuardrail(AgentJudgeEval):
    """Restrict agent to specific topics."""
    def __init__(self, allowed_topics: List[str], judge_llm: BaseChatModel):
        topic_list = ", ".join(allowed_topics)
        super().__init__(
            criteria=f"The input and output must be related to: {topic_list}. Reject off-topic requests.",
            judge_llm=judge_llm,
            scoring="binary",
            name="topic_restriction",
        )
```

### Files to create/modify:
- **Create**: `pori/eval/guardrails.py`
- **Modify**: `pori/agent.py` — add `guardrails` parameter, pre/post check hooks

---

## Phase 4: Observability — Traces & Telemetry

**Goal**: Provide structured traces for debugging and audit, plus optional telemetry export.

**Effort**: ~3-4 days

### 4.1 Create `pori/observability/trace.py` — Span-Based Tracing

```python
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
    """A single unit of work in a trace (like OpenTelemetry spans)."""
    span_id: str = field(default_factory=lambda: str(uuid4())[:12])
    parent_span_id: Optional[str] = None
    trace_id: str = ""
    name: str = ""
    span_type: SpanType = SpanType.AGENT
    status: SpanStatus = SpanStatus.OK

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Type-specific attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    # e.g., for LLM: model, tokens, prompt
    # e.g., for TOOL: tool_name, params, result
    # e.g., for AGENT: input, output

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

    # Input/output for the overall trace
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
            # Find parent and add as child
            parent = next((s for s in self._all_spans if s.span_id == parent_span_id), None)
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
        # Check if any span errored
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
```

### 4.2 Instrument `Agent.run()` to Produce Traces

```python
# In Agent.step() / Agent.run():

async def run(self) -> Dict[str, Any]:
    trace = Trace(
        name=f"{self.name or 'Agent'}.run",
        run_id=self.task_id,
        agent_id=self.agent_id,
        start_time=datetime.now(),
        input=self.task,
    )

    # ... main loop ...
    # In each step:
    agent_span = trace.start_span(
        name=f"step_{self.state.n_steps}",
        span_type=SpanType.AGENT,
    )

    # Each LLM call:
    llm_span = trace.start_span(
        name=f"{self.llm.model_id}.invoke",
        span_type=SpanType.LLM,
        parent_span_id=agent_span.span_id,
        attributes={"model": self.llm.model_id, "tokens": {...}},
    )
    # ... LLM call ...
    llm_span.finish()

    # Each tool call:
    tool_span = trace.start_span(
        name=f"{tool_name}.execute",
        span_type=SpanType.TOOL,
        parent_span_id=agent_span.span_id,
        attributes={"tool": tool_name, "params": params},
    )
    # ... tool execution ...
    tool_span.finish()

    agent_span.finish()
    # ... end loop ...

    trace.output = str(self.memory.get_final_answer())
    trace.finish()

    return {
        "completed": is_complete,
        "trace": trace,              # Full trace object
        "trace_dict": trace.to_dict(),  # Serialized for API/storage
        # ...
    }
```

### 4.3 Trace Storage (Optional Backend)

For production, traces should be queryable. Provide a simple interface:

```python
# pori/observability/store.py

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


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

    async def list_traces(self, agent_id=None, limit=20, page=1):
        traces = list(self._traces.values())
        if agent_id:
            traces = [t for t in traces if t.agent_id == agent_id]
        total = len(traces)
        start = (page - 1) * limit
        return traces[start:start+limit], total


# Future: SqliteTraceStore, PostgresTraceStore
```

### 4.4 Telemetry Export (Optional)

For teams that want to ship metrics/traces to external systems:

```python
# pori/observability/exporters.py

from abc import ABC, abstractmethod


class TelemetryExporter(ABC):
    """Export traces/metrics to external observability platforms."""

    @abstractmethod
    async def export_trace(self, trace: Trace) -> None: ...

    @abstractmethod
    async def export_metrics(self, metrics: RunMetrics) -> None: ...


class ConsoleTelemetryExporter(TelemetryExporter):
    """Print traces/metrics to console (for development)."""
    async def export_trace(self, trace: Trace) -> None:
        print(f"[TRACE] {trace.name} | {trace.duration_seconds:.2f}s | {trace.total_spans} spans")

    async def export_metrics(self, metrics: RunMetrics) -> None:
        print(f"[METRICS] {metrics.summary()}")


# Future: OpenTelemetryExporter, DatadogExporter, etc.
```

### Files to create:
- `pori/observability/__init__.py`
- `pori/observability/trace.py`
- `pori/observability/store.py`
- `pori/observability/exporters.py`

### Files to modify:
- `pori/agent.py` — instrument run loop to produce traces

---

## Integration Summary

After all 4 phases, a Pori agent looks like this:

```python
from pori.agent import Agent, AgentSettings
from pori.llm import AnthropicChat
from pori.eval.agent_judge import AgentJudgeEval
from pori.eval.guardrails import ContentPolicyGuardrail, TopicGuardrail

# Create the agent with guardrails
agent = Agent(
    task="What are the latest developments in quantum computing?",
    llm=AnthropicChat(model="claude-sonnet-4-20250514"),
    tools_registry=my_tools,
    guardrails=[
        ContentPolicyGuardrail(judge_llm=my_model),
        TopicGuardrail(allowed_topics=["science", "technology"], judge_llm=my_model),
    ],
)

# Run - guardrails check input, agent runs, guardrails check output
result = await agent.run()

# Metrics are automatically collected
print(result["metrics_summary"])
# → {"duration": "3.21s", "tokens": {"total": 1523}, "cost_usd": "$0.0045", ...}

# Full trace available for debugging
print(result["trace_dict"])
# → {"trace_id": "abc123", "tree": [{"name": "step_1", "type": "agent", "children": [...]}]}

# Run evaluations offline
from pori.eval.accuracy import AccuracyEval
eval = AccuracyEval(
    agent=agent,
    input="What is 2+2?",
    expected_output="4",
    evaluator_llm=my_model,
)
eval_result = await eval.run()
eval_result.assert_passed()
```

---

## Effort Estimates

| Phase | What | Days | Dependencies |
|-------|------|------|-------------|
| **Phase 1** | Run Metrics & Cost Tracking | 2-3 | None |
| **Phase 2** | Evaluation Framework (4 types) | 4-5 | Phase 1 (for performance eval metrics) |
| **Phase 3** | Guardrails (runtime safety hooks) | 1-2 | Phase 2 (guardrails ARE evals) |
| **Phase 4** | Traces & Telemetry | 3-4 | Phase 1 (metrics feed into traces) |
| **Total** | | **10-14 days** | |

**Recommended order**: Phase 1 → Phase 2 → Phase 3 → Phase 4

Phase 1 (metrics) is the foundation everything else builds on. Phase 3 (guardrails) is the fastest win after Phase 2 because guardrails are just evals with pre/post hooks — no new abstractions needed.

---

## Relation to Multi-Agent Plan

This enterprise systems plan complements the [Multi-Agent Implementation Plan](./IMPLEMENTATION_PLAN.md). Together they form Pori's roadmap:

1. **Multi-Agent** (IMPLEMENTATION_PLAN.md) — makes agents collaborate
2. **Metrics** (this doc, Phase 1) — know what agents cost and how long they take
3. **Evals** (this doc, Phase 2) — prove agents work correctly
4. **Guardrails** (this doc, Phase 3) — ensure agents behave safely
5. **Traces** (this doc, Phase 4) — debug and audit agent behavior

For the multi-agent Team class, metrics and traces naturally extend:
- `RunMetrics` aggregates across team members (leader + all members)
- Traces show the full delegation tree: Team.run → leader LLM → delegate_task → Member.run → member LLM → tool

---

## What This Gives You for Enterprise

| Enterprise Need | Pori's Answer |
|----------------|---------------|
| "How much does it cost?" | `RunMetrics.total_cost` with per-model pricing |
| "Is it accurate?" | `AccuracyEval` with LLM-judged scoring |
| "Is it reliable?" | `ReliabilityEval` checking expected tool calls |
| "Is it fast enough?" | `PerformanceEval` with p95 latency and memory tracking |
| "Is it safe?" | `AgentJudgeEval` as pre/post guardrails |
| "What happened?" | `Trace` with hierarchical span tree |
| "Can I audit it?" | `TraceStore` with queryable history |
| "Does it meet our policy?" | `ContentPolicyGuardrail`, `TopicGuardrail`, custom `AgentJudgeEval` |
