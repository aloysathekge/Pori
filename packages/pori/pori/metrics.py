from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

# $ per MTok (million tokens): (input, output). Cache tiers not included.
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    # OpenAI — GPT-5 (future)
    "gpt-5.2": (1.75, 14.00),
    "gpt-5-2": (1.75, 14.00),
    "gpt-5.2-pro": (21.00, 168.00),
    "gpt-5-2-pro": (21.00, 168.00),
    "gpt-5-mini": (0.25, 2.00),
    # OpenAI — GPT-4
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # Anthropic — Claude
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-sonnet-3-7": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-3-5": (0.80, 4.00),
    "claude-haiku-3": (0.25, 1.25),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-opus-4-1": (15.00, 75.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-opus-3": (15.00, 75.00),
}


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
        cost = self.total_cost
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
            "tokens_raw": tokens,
            "cost_usd": f"${cost:.4f}" if cost else None,
        }


def estimate_llm_call_cost(model_id: str, tokens: TokenUsage) -> Optional[float]:
    """Estimate dollar cost for one LLM call based on token usage."""
    if not model_id:
        return None
    prices = PRICE_PER_MTOK.get(model_id)
    if not prices:
        return None
    in_price_per_mtok, out_price_per_mtok = prices
    in_cost = (tokens.input_tokens / 1_000_000) * in_price_per_mtok
    out_cost = (tokens.output_tokens / 1_000_000) * out_price_per_mtok
    return in_cost + out_cost
