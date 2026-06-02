"""Tests for the shared LLM retry helper (pori.llm.retry)."""

import asyncio

import pytest

from pori.llm.retry import RetryConfig, is_transient_error, retry_async

pytestmark = pytest.mark.unit

# Zero-delay config so tests never actually sleep.
NO_DELAY = RetryConfig(max_retries=3, base_delay=0.0, max_delay=0.0)


# Exceptions named like the real SDK transient errors (matched by class name).
class RateLimitError(Exception):
    pass


class APITimeoutError(Exception):
    pass


class _StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _RetryAfterError(Exception):
    # Mirrors a real rate-limit error: transient status + a Retry-After hint.
    def __init__(self, retry_after):
        super().__init__("slow down")
        self.status_code = 429
        self.retry_after = retry_after


def test_transient_detection_by_name():
    assert is_transient_error(RateLimitError("429"))
    assert is_transient_error(APITimeoutError("timeout"))


def test_transient_detection_by_status_code():
    assert is_transient_error(_StatusError(503))
    assert is_transient_error(_StatusError(429))
    assert not is_transient_error(_StatusError(400))
    assert not is_transient_error(_StatusError(401))


def test_transient_detection_builtins():
    assert is_transient_error(asyncio.TimeoutError())
    assert is_transient_error(ConnectionError())


def test_permanent_errors_not_transient():
    assert not is_transient_error(ValueError("bad schema"))
    assert not is_transient_error(KeyError("missing"))


async def test_retries_then_succeeds():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError("429")
        return "ok"

    result = await retry_async(flaky, NO_DELAY, label="test")
    assert result == "ok"
    assert calls["n"] == 3  # failed twice, succeeded on third


async def test_permanent_error_not_retried():
    calls = {"n": 0}

    async def boom():
        calls["n"] += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        await retry_async(boom, NO_DELAY, label="test")
    assert calls["n"] == 1  # no retries on permanent error


async def test_exhausts_retries_then_raises():
    calls = {"n": 0}

    async def always_429():
        calls["n"] += 1
        raise RateLimitError("429")

    with pytest.raises(RateLimitError):
        await retry_async(
            always_429,
            RetryConfig(max_retries=2, base_delay=0.0, max_delay=0.0),
            label="test",
        )
    # initial attempt + 2 retries == 3 calls
    assert calls["n"] == 3


async def test_zero_retries_means_single_attempt():
    calls = {"n": 0}

    async def always_fail():
        calls["n"] += 1
        raise APITimeoutError("timeout")

    with pytest.raises(APITimeoutError):
        await retry_async(always_fail, RetryConfig(max_retries=0), label="test")
    assert calls["n"] == 1


async def test_retry_after_is_honored_but_capped():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            # retry_after of 100s, but max_delay caps it to 0 so the test is instant
            raise _RetryAfterError(retry_after=100)
        return "done"

    result = await retry_async(flaky, NO_DELAY, label="test")
    assert result == "done"
    assert calls["n"] == 2


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("PORI_LLM_MAX_RETRIES", "5")
    monkeypatch.setenv("PORI_LLM_RETRY_BASE_DELAY", "1.5")
    monkeypatch.setenv("PORI_LLM_RETRY_MAX_DELAY", "20")
    cfg = RetryConfig.from_env()
    assert cfg.max_retries == 5
    assert cfg.base_delay == 1.5
    assert cfg.max_delay == 20.0


def test_config_from_env_handles_garbage(monkeypatch):
    monkeypatch.setenv("PORI_LLM_MAX_RETRIES", "not-a-number")
    cfg = RetryConfig.from_env()
    assert cfg.max_retries == 2  # falls back to default
