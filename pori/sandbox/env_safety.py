"""Keep secrets out of agent-run subprocesses (Hermes pattern).

The agent's shell commands must never see the host's provider credentials:
even an innocent `printenv` would spill every API key into the transcript.
``sanitized_subprocess_env()`` returns a copy of the environment with
anything secret-shaped removed — applied by the local sandbox before every
command, and the same discipline applies to remote backends (which get a
fresh VM env by construction).
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional

# Secret-shaped names: strip by pattern rather than enumerating providers,
# so a newly added key can't leak just because nobody updated a list.
_SECRET_PATTERN = re.compile(
    r"(API_?KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIALS?|PRIVATE_KEY|AUTH)",
    re.IGNORECASE,
)

# Names that don't match the pattern but must never be inherited.
_ALWAYS_STRIP = frozenset(
    {
        "DATABASE_URL",
        "SUPABASE_URL",
        # Active-environment markers: inheriting them makes agent commands
        # silently operate on the *host's* venv/conda env (Hermes local.py).
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
    }
)

# Pattern-matching names that are safe and commonly needed (allowlist wins).
_ALLOW = frozenset({"PATH", "PATHEXT", "AUTHORITY", "XDG_SESSION_TYPE"})


def sanitized_subprocess_env(
    base: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """A copy of the environment safe to hand to agent-run commands."""
    source = dict(base if base is not None else os.environ)
    safe: Dict[str, str] = {}
    for key, value in source.items():
        upper = key.upper()
        if upper in _ALLOW:
            safe[key] = value
            continue
        if upper in _ALWAYS_STRIP or _SECRET_PATTERN.search(upper):
            continue
        safe[key] = value
    return safe


__all__ = ["sanitized_subprocess_env"]
