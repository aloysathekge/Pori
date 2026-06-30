"""Cache-tiered system-prompt assembly.

The system prompt is built from three ordered tiers and joined `stable ->
context -> volatile`, so the cacheable prefix (identity, operating rules, tool
guidance) stays warm across turns while per-run content (skills now; memory and
the live plan in a later phase) lives at the end.

See `.agent/reference-studies/system-prompt-architecture.md`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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


_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# Shipped, comments-only persona template (yields no persona -> default identity).
_SHIPPED_SOUL = Path(__file__).parent / "system" / "SOUL.md"


def _read_persona(path: Path) -> str:
    """Return the persona text from a SOUL.md (comments + headings stripped)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    text = _HTML_COMMENT.sub("", text)
    has_persona = any(
        line.strip() and not line.strip().startswith("#") for line in text.splitlines()
    )
    return text.strip() if has_persona else ""


def resolve_identity(
    soul_path: Optional[str] = None,
    cwd: Optional[Path] = None,
) -> str:
    """Resolve the agent identity: a user SOUL.md persona, else DEFAULT_IDENTITY.

    Resolution order (first with real content wins): project ``./SOUL.md`` ->
    ``soul_path`` -> shipped template (comments-only). The file is read fresh on
    each call, so editing it takes effect on the next task without a restart.
    """
    base = Path(cwd) if cwd else Path.cwd()
    candidates = [base / "SOUL.md"]
    if soul_path:
        candidates.append(Path(soul_path).expanduser())
    candidates.append(_SHIPPED_SOUL)
    for path in candidates:
        if path.is_file():
            persona = _read_persona(path)
            if persona:
                return persona
    return DEFAULT_IDENTITY


__all__ = [
    "DEFAULT_IDENTITY",
    "SystemPromptTiers",
    "build_system_prompt",
    "resolve_identity",
]
