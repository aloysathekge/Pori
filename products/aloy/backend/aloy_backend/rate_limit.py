"""In-memory sliding window rate limiter per user."""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Depends, HTTPException, status

from .config import settings
from .tenancy import OrganizationContext, Permission, get_organization_context

_RATE_LIMITED_PERMISSIONS = frozenset(
    {
        Permission.AGENT_WRITE,
        Permission.MEMORY_WRITE,
        Permission.RUN_CREATE,
    }
)


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


def rate_limited_permission(permission: Permission):
    """Authorize and throttle work-creating or user-state mutations.

    Read-only routes and cancellation must use ``require_permission`` instead.
    Live projections legitimately issue many reads, while stop controls must
    remain available under load. Keeping that boundary executable prevents UI
    refresh traffic from consuming the scarce mutation/Run budget.
    """
    if permission not in _RATE_LIMITED_PERMISSIONS:
        raise ValueError(
            f"Permission {permission.value} must use require_permission; "
            "read and cancellation traffic is not part of the mutation rate limit"
        )

    async def dependency(
        context: OrganizationContext = Depends(get_organization_context),
    ) -> OrganizationContext:
        if not context.permits(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permission: {permission.value}",
            )
        _limiter.check(f"{context.organization_id}:{context.user_id}")
        return context

    return dependency
