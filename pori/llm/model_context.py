"""Model context-length lookup for model-aware context sizing.

Pori sizes its conversation-history budget to the model's real context length
instead of a fixed number, so a 1M-context model actually uses its capacity and
a small local model stays bounded — with compression (AC-3) as the safety net for
runs that overflow even a large window. The length is matched by substring
against the model id; unknown models fall back to a conservative default.
"""

from __future__ import annotations

from typing import List, Tuple

DEFAULT_CONTEXT_LENGTH = 128_000

# (substring, context_length) — checked in order, first match wins, so more
# specific markers must precede more general ones.
_MODEL_CONTEXT_LENGTHS: List[Tuple[str, int]] = [
    # Google Gemini
    ("gemini-1.5-pro", 2_000_000),
    ("gemini-1.5-flash", 1_000_000),
    ("gemini-2.5-pro", 1_000_000),
    ("gemini-2", 1_000_000),
    ("gemini", 1_000_000),
    # OpenAI
    ("gpt-4.1", 1_000_000),
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("o1", 200_000),
    ("o3", 200_000),
    ("o4", 200_000),
    ("gpt-4", 128_000),
    ("gpt-3.5", 16_385),
    # Anthropic Claude (200K standard window; 1M-beta not assumed)
    ("claude", 200_000),
    # Common open / hosted models
    ("deepseek", 128_000),
    ("llama", 128_000),
    ("mixtral", 32_000),
    ("mistral", 32_000),
    ("qwen", 128_000),
    ("command-r", 128_000),
]


def get_model_context_length(model: str, default: int = DEFAULT_CONTEXT_LENGTH) -> int:
    """Return the context length (in tokens) for a model id, or ``default``."""
    if not model:
        return default
    m = model.lower()
    for marker, length in _MODEL_CONTEXT_LENGTHS:
        if marker in m:
            return length
    return default
