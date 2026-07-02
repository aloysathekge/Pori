"""Hardline command safety floor (INF-1).

A tiny, **non-bypassable** denylist of shell commands that can irrecoverably
destroy the host. It is checked inside the sandbox's command execution — below
any human-in-the-loop approval — so these commands can never be "approved away".

Deliberately kept small and *no-recovery-only*: recoverable-but-scary operations
(e.g. deleting a project subdir) are what HITL is for, and a large list would
cause false positives that break legitimate work. Detection runs after
obfuscation-normalization (NFKC fold, strip null bytes, backslash-escapes, and
empty-string token splits) so trivial evasions like ``r''m -rf /`` or
``rm\\ -rf\\ /`` don't slip through.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Pattern

# A command starts at the beginning, after a separator (; | &), or after sudo.
_CMD = r"(?:^|[\n;|&]\s*|\bsudo\s+)"

_HARDLINE_PATTERNS: List[Pattern[str]] = [
    # rm with a recursive flag whose target is root or the whole home dir.
    # Two order-independent lookaheads: a recursive flag AND a root/home target,
    # both before the next command separator. A subdir target (rm -rf ./build,
    # rm -rf /tmp/x) is NOT matched — that's HITL's job.
    re.compile(
        _CMD + r"rm\b"
        r"(?=[^;|&\n]*?(?:-[a-z]*r[a-z]*|--recursive))"
        r"(?=[^;|&\n]*?\s(?:/(?=[\s*]|$)|~/?(?=\s|$)|\$home\b|/\*))",
        re.IGNORECASE,
    ),
    # rm with root protection disabled — always catastrophic intent.
    re.compile(r"--no-preserve-root", re.IGNORECASE),
    # Format a filesystem.
    re.compile(_CMD + r"mkfs(?:\.\w+)?\b", re.IGNORECASE),
    # Write raw bytes over a block device.
    re.compile(
        _CMD + r"dd\b[^;|&\n]*\bof=\s*/dev/(?:sd|nvme|hd|disk|xvd|mapper)",
        re.IGNORECASE,
    ),
    # Redirect straight into a raw block device.
    re.compile(r">\s*/dev/(?:sd|nvme|hd|disk|xvd)\w*", re.IGNORECASE),
    # Classic fork bomb :(){ :|:& };:
    re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.IGNORECASE),
    # Power-state changes.
    re.compile(_CMD + r"(?:shutdown|reboot|halt|poweroff)\b", re.IGNORECASE),
    # kill -1 (signal every process the user can reach).
    re.compile(_CMD + r"kill\b[^;|&\n]*\s-(?:9\s+)?-?1\b", re.IGNORECASE),
]


def normalize(command: str) -> str:
    """Normalize a command for detection (NFKC, de-obfuscation, whitespace fold)."""
    if not command:
        return ""
    text = unicodedata.normalize("NFKC", command)
    text = text.replace("\x00", "")
    # Empty-string token splits used to break up keywords: rm'' -> rm, r""m -> rm.
    text = text.replace("''", "").replace('""', "")
    # Backslash escapes: rm\ -rf -> rm -rf, r\m -> rm.
    text = re.sub(r"\\(.)", r"\1", text)
    # Collapse whitespace so separators/flags match regardless of spacing.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def hardline_violation(command: str) -> Optional[str]:
    """Return a short reason if the command hits the hardline floor, else None."""
    norm = normalize(command)
    for pat in _HARDLINE_PATTERNS:
        if pat.search(norm):
            return "matched a hardline (irrecoverable) command pattern"
    return None
