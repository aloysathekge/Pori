from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


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
    cost: Optional[float] = None
    duration_seconds: float = 0.0
    time_to_first_token: Optional[float] = None
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

