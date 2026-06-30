"""Cache-tiered system-prompt assembly.

The system prompt is built from three ordered tiers and joined `stable ->
context -> volatile`, so the cacheable prefix (identity, operating rules, tool
guidance) stays warm across turns while per-run content (skills now; memory and
the live plan in a later phase) lives at the end.

See `.agent/reference-studies/system-prompt-architecture.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# Neutral, framework-default identity (Hermes-faithful). It is the fallback
# "who am I" block; a user-supplied persona via SOUL.md will override it in a
# later phase. Behavioral commitments stay in the operating-rule blocks, not
# here, so they hold regardless of any user persona.
DEFAULT_IDENTITY = (
    "You are Pori, an open-source AI agent. You are helpful, knowledgeable, and "
    "direct. You assist with answering questions, writing and editing code, "
    "analyzing information, and executing actions through your tools. You "
    "communicate clearly, admit uncertainty when appropriate, and prioritize "
    "being genuinely useful over being verbose. You are targeted and efficient "
    "in exploration. When asked who you are, answer plainly and briefly (e.g. "
    '"I\'m Pori, an AI assistant. How can I help?") and never refuse or deflect '
    "a self-introduction."
)


@dataclass
class SystemPromptTiers:
    """Ordered prompt tiers, rendered `stable -> context -> volatile`.

    - stable:   identity + operating rules + tool guidance (cache-warm).
    - context:  caller/project content (custom prompt; project files later).
    - volatile: per-run content (available + selected skills; memory/plan later).
    """

    stable: List[str] = field(default_factory=list)
    context: List[str] = field(default_factory=list)
    volatile: List[str] = field(default_factory=list)

    def blocks(self) -> List[str]:
        ordered = (*self.stable, *self.context, *self.volatile)
        return [block.strip() for block in ordered if block and block.strip()]

    def render(self) -> str:
        return "\n\n".join(self.blocks())


def build_system_prompt(tiers: SystemPromptTiers) -> str:
    """Render ordered prompt tiers into a single system prompt string."""
    return tiers.render()


__all__ = ["DEFAULT_IDENTITY", "SystemPromptTiers", "build_system_prompt"]
