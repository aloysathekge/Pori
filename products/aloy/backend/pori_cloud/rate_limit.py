"""In-memory sliding window rate limiter per user."""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Depends, HTTPException, status

from .config import settings
from .tenancy import OrganizationContext, get_organization_context


class RateLimiter:
    """Sliding window rate limiter. Tracks request timestamps per user."""

    def __init__(self, rpm: int = 60):
        self.rpm = rpm
        self._windows: dict[str, list[float]] = defaultdict(list)

    def check(self, user_id: str) -> None:
        now = time.monotonic()
        window_start = now - 60.0

        # Drop timestamps outside the window
        timestamps = self._windows[user_id]
        self._windows[user_id] = [t for t in timestamps if t > window_start]

        if len(self._windows[user_id]) >= self.rpm:
            retry_after = int(60 - (now - self._windows[user_id][0])) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Max {self.rpm} requests per minute.",
                headers={"Retry-After": str(max(1, retry_after))},
            )

        self._windows[user_id].append(now)


# Singleton instance
_limiter = RateLimiter(rpm=settings.rate_limit_rpm)


async def check_rate_limit(
    context: OrganizationContext = Depends(get_organization_context),
) -> OrganizationContext:
    """Enforce rate limiting within the active organization and user scope."""
    _limiter.check(f"{context.organization_id}:{context.user_id}")
    return context
