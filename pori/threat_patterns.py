"""Deterministic prompt-injection / exfiltration scanner (INF-5).

A cheap, LLM-free regex scan of *untrusted* content — tool results and text being
written into memory (content the user did not author). It is the deterministic
counterpart to the LLM-judged guardrails: an LLM judge is a poor fit for injection
detection (it is the very thing being manipulated) and burns tokens. Anchored on
attack *vocabulary* (instruction-override, secret exfiltration, invisible Unicode)
rather than "bossy English", so it stays low-false-positive.

Policy split (mirrors the Hermes design): **warn** on tool results (annotate, let
the model see it), **block** on memory writes (refuse to persist).
"""

from __future__ import annotations

import re
from typing import List, Optional, Pattern

# Zero-width, bidi, and BOM characters used to smuggle hidden instructions.
# Built from integer codepoints so the source stays pure-ASCII and readable.
INVISIBLE_CHARS = frozenset(
    chr(cp)
    for cp in (
        0x200B,  # zero-width space
        0x200C,  # zero-width non-joiner
        0x200D,  # zero-width joiner
        0x200E,  # left-to-right mark
        0x200F,  # right-to-left mark
        0x202A,  # left-to-right embedding
        0x202B,  # right-to-left embedding
        0x202C,  # pop directional formatting
        0x202D,  # left-to-right override
        0x202E,  # right-to-left override
        0x2060,  # word joiner
        0x2061,  # function application
        0x2062,  # invisible times
        0x2063,  # invisible separator
        0x2064,  # invisible plus
        0xFEFF,  # BOM / zero-width no-break space
    )
)
# Unicode Tags block (U+E0000-U+E007F) — used for hidden-text attacks.
_TAG_RANGE = re.compile(r"[\U000E0000-\U000E007F]")

_THREAT_PATTERNS: List[Pattern[str]] = [
    # Instruction override / jailbreak vocabulary.
    re.compile(
        r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
        r"(?:instructions?|prompts?|messages?|context)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdisregard\s+(?:all\s+|any\s+)?(?:previous|prior|the\s+above|your)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bforget\s+(?:everything|all\s+(?:previous|prior)|your\s+instructions)\b",
        re.IGNORECASE,
    ),
    # Fake system/instruction framing injected into untrusted text.
    re.compile(r"<\s*/?\s*(?:system|instructions?)\s*>", re.IGNORECASE),
    re.compile(
        r"\b(?:system|developer)\s+(?:prompt|message|override)\s*[:=]", re.IGNORECASE
    ),
    # Secret exfiltration — the credential nouns are kept specific (not bare
    # "secret"/"token") so benign text like "print the secret sauce" is not hit.
    re.compile(
        r"\b(?:curl|wget|fetch|invoke-webrequest)\b[^\n]*"
        r"(?:api[-_\s]?key|access[-_\s]?token|auth[-_\s]?token|password|credential|"
        r"private[-_\s]?key|secret[-_\s]?key)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:send|post|exfiltrate|leak|upload|email|reveal)\b[^\n]*\b"
        r"(?:api[-_\s]?keys?|access[-_\s]?tokens?|auth[-_\s]?tokens?|passwords?|"
        r"credentials?|private[-_\s]?keys?|secret[-_\s]?keys?)",
        re.IGNORECASE,
    ),
]


def _has_invisible(text: str) -> bool:
    return any(c in INVISIBLE_CHARS for c in text) or bool(_TAG_RANGE.search(text))


def scan_for_threats(content: str, scope: str = "all") -> List[str]:
    """Return a list of threat reasons found in ``content`` (empty if clean)."""
    if not content:
        return []
    reasons: List[str] = []
    if _has_invisible(content):
        reasons.append("hidden/invisible Unicode characters")
    for pat in _THREAT_PATTERNS:
        if pat.search(content):
            reasons.append("instruction-override or exfiltration phrasing")
            break
    return reasons


def first_threat_message(content: str, scope: str = "all") -> Optional[str]:
    """Return one human-readable threat message, or None if ``content`` is clean."""
    reasons = scan_for_threats(content, scope=scope)
    if not reasons:
        return None
    return "Potential prompt injection / exfiltration (" + "; ".join(reasons) + ")"
