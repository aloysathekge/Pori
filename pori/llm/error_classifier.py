"""Classify LLM provider errors into actionable failure reasons (AC-3... AC-2).

A generic "is this transient?" check can't tell a context-overflow 400 (retrying
unchanged is futile — compress instead) from a billing 402 (stop, don't burn
retries) from a rate-limit 429 (back off and retry). This module walks the
exception cause-chain, pulls a status code + message, and maps them to a
:class:`FailoverReason` with precomputed recovery hints the caller acts on:

- ``retryable`` — safe to retry with backoff (rate limit, overload, 5xx, timeout).
- ``should_compress`` — the request is too big; shrink/compress and retry once.
- ``should_fail_fast`` — hopeless without user action (auth, billing); stop now.

Provider-agnostic: matched by exception class name and HTTP status, no SDK import.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FailoverReason(str, Enum):
    AUTH = "auth"
    BILLING = "billing"
    RATE_LIMIT = "rate_limit"
    OVERLOADED = "overloaded"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    CONTEXT_OVERFLOW = "context_overflow"
    CONTENT_POLICY_BLOCKED = "content_policy_blocked"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClassifiedError:
    reason: FailoverReason
    retryable: bool
    should_compress: bool = False
    should_fail_fast: bool = False
    message: str = ""


# reason -> (retryable, should_compress, should_fail_fast)
_HINTS: dict[FailoverReason, tuple[bool, bool, bool]] = {
    FailoverReason.RATE_LIMIT: (True, False, False),
    FailoverReason.OVERLOADED: (True, False, False),
    FailoverReason.SERVER_ERROR: (True, False, False),
    FailoverReason.TIMEOUT: (True, False, False),
    FailoverReason.CONTEXT_OVERFLOW: (False, True, False),
    FailoverReason.AUTH: (False, False, True),
    FailoverReason.BILLING: (False, False, True),
    FailoverReason.CONTENT_POLICY_BLOCKED: (False, False, False),
    FailoverReason.UNKNOWN: (False, False, False),
}

_RATE_LIMIT_NAMES = {"RateLimitError", "ResourceExhausted", "TryAgain"}
_OVERLOADED_NAMES = {"OverloadedError"}
_TIMEOUT_NAMES = {
    "APITimeoutError",
    "APIConnectionError",
    "APIConnectionTimeoutError",
    "DeadlineExceeded",
    "Timeout",
}
_SERVER_NAMES = {
    "InternalServerError",
    "ServiceUnavailableError",
    "ServiceUnavailable",
    "ServerError",
}
_SERVER_STATUS = {408, 409, 425, 500, 502, 503, 504, 529}
_OVERFLOW_MARKERS = (
    "too long",
    "maximum context",
    "context length",
    "context_length_exceeded",
    "prompt is too long",
    "too many tokens",
    "reduce the length",
)
_POLICY_MARKERS = ("content policy", "content_filter", "safety", "blocked by")


def _extract_status_code(exc: BaseException) -> Optional[int]:
    """Walk the cause chain for an HTTP status code."""
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        for attr in ("status_code", "code", "status"):
            val = getattr(cur, attr, None)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass
        resp = getattr(cur, "response", None)
        if resp is not None:
            val = getattr(resp, "status_code", None)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass
        cur = cur.__cause__ or cur.__context__
    return None


def _extract_message(exc: BaseException) -> str:
    """Concatenate the messages along the cause chain (lower-cased)."""
    parts: list[str] = []
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(str(cur))
        cur = cur.__cause__ or cur.__context__
    return " ".join(parts).lower()


def _mk(reason: FailoverReason, message: str) -> ClassifiedError:
    retryable, should_compress, should_fail_fast = _HINTS[reason]
    return ClassifiedError(
        reason=reason,
        retryable=retryable,
        should_compress=should_compress,
        should_fail_fast=should_fail_fast,
        message=message,
    )


def classify_error(exc: BaseException) -> ClassifiedError:
    """Map a provider exception to a :class:`ClassifiedError` with recovery hints."""
    name = type(exc).__name__
    message = _extract_message(exc)

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return _mk(FailoverReason.TIMEOUT, message)
    if name in _RATE_LIMIT_NAMES:
        return _mk(FailoverReason.RATE_LIMIT, message)
    if name in _OVERLOADED_NAMES:
        return _mk(FailoverReason.OVERLOADED, message)
    if name in _TIMEOUT_NAMES:
        return _mk(FailoverReason.TIMEOUT, message)
    if name in _SERVER_NAMES:
        return _mk(FailoverReason.SERVER_ERROR, message)

    status = _extract_status_code(exc)
    if status == 429:
        return _mk(FailoverReason.RATE_LIMIT, message)
    if status in (401, 403):
        return _mk(FailoverReason.AUTH, message)
    if status == 402:
        return _mk(FailoverReason.BILLING, message)
    if status == 400:
        if any(m in message for m in _OVERFLOW_MARKERS):
            return _mk(FailoverReason.CONTEXT_OVERFLOW, message)
        if any(m in message for m in _POLICY_MARKERS):
            return _mk(FailoverReason.CONTENT_POLICY_BLOCKED, message)
        return _mk(FailoverReason.UNKNOWN, message)
    if status in _SERVER_STATUS:
        return _mk(FailoverReason.SERVER_ERROR, message)

    # Some SDKs raise a bare BadRequest without a status attribute.
    if any(m in message for m in _OVERFLOW_MARKERS):
        return _mk(FailoverReason.CONTEXT_OVERFLOW, message)

    return _mk(FailoverReason.UNKNOWN, message)
