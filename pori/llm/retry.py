"""Resilient retry helper for LLM provider network calls.

Wraps provider API calls in bounded exponential backoff with full jitter so a
single transient hiccup (rate limit, timeout, dropped connection, 5xx) doesn't
fail a whole agent step. Permanent errors (auth, bad request, schema/validation)
are never retried — they re-raise immediately.

The helper deliberately imports no provider SDKs: transient errors are detected
by exception class name and HTTP status code, which keeps it shared across the
Anthropic, OpenAI/OpenRouter/Fireworks, and Google wrappers.

Configurable via environment variables:
    PORI_LLM_MAX_RETRIES      (default 2 → up to 3 attempts total)
    PORI_LLM_RETRY_BASE_DELAY (default 0.5 seconds)
    PORI_LLM_RETRY_MAX_DELAY  (default 8.0 seconds)
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, TypeVar

from .error_classifier import classify_error

logger = logging.getLogger("pori.llm.retry")

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Bounded-backoff retry settings."""

    max_retries: int = 2
    base_delay: float = 0.5
    max_delay: float = 8.0

    @classmethod
    def from_env(cls) -> "RetryConfig":
        def _int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except (TypeError, ValueError):
                return default

        def _float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except (TypeError, ValueError):
                return default

        return cls(
            max_retries=max(0, _int("PORI_LLM_MAX_RETRIES", 2)),
            base_delay=max(0.0, _float("PORI_LLM_RETRY_BASE_DELAY", 0.5)),
            max_delay=max(0.0, _float("PORI_LLM_RETRY_MAX_DELAY", 8.0)),
        )


def is_transient_error(exc: BaseException) -> bool:
    """Return True if the error is worth retrying.

    Backed by :func:`pori.llm.error_classifier.classify_error`, so retry decisions
    stay consistent with the richer recovery logic in the agent loop (which also
    distinguishes context-overflow and auth/billing failures).
    """
    return classify_error(exc).retryable


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Honor a server-provided Retry-After hint when present."""
    ra = getattr(exc, "retry_after", None)
    if ra is not None:
        try:
            return float(ra)
        except (TypeError, ValueError):
            pass
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers is not None:
        try:
            val = headers.get("retry-after") or headers.get("Retry-After")
            if val is not None:
                return float(val)
        except (TypeError, ValueError, AttributeError):
            pass
    return None


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    config: Optional[RetryConfig] = None,
    *,
    label: str = "llm",
) -> T:
    """Run an async call with bounded exponential backoff on transient errors.

    Args:
        fn: A zero-arg coroutine factory performing the actual API call.
        config: Retry settings; falls back to environment defaults if None.
        label: Short tag for log lines (e.g. the provider name).
    """
    cfg = config or RetryConfig.from_env()
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 — re-raised below unless transient
            if attempt >= cfg.max_retries or not is_transient_error(exc):
                raise
            server_delay = _retry_after_seconds(exc)
            if server_delay is not None:
                delay = min(server_delay, cfg.max_delay)
            else:
                ceiling = min(cfg.max_delay, cfg.base_delay * (2**attempt))
                # Full jitter avoids synchronized retry storms across agents.
                delay = random.uniform(0, ceiling) if ceiling > 0 else 0.0
            attempt += 1
            logger.warning(
                "Transient %s error (%s): retry %d/%d in %.2fs",
                label,
                type(exc).__name__,
                attempt,
                cfg.max_retries,
                delay,
            )
            await asyncio.sleep(delay)
