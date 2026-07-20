from __future__ import annotations

import pytest
from fastapi import HTTPException

import aloy_backend.rate_limit as rate_limit_module
from aloy_backend.rate_limit import RateLimiter, rate_limited_permission
from aloy_backend.routes import event_memory, event_setup, events, surfaces, today
from aloy_backend.tenancy import (
    OrganizationContext,
    OrganizationPolicy,
    Permission,
    require_permission,
)


def _context(*permissions: Permission) -> OrganizationContext:
    return OrganizationContext(
        organization_id="org-rate-limit",
        user_id="user-rate-limit",
        role="member",
        permissions=tuple(permission.value for permission in permissions),
        policy=OrganizationPolicy(),
    )


@pytest.mark.parametrize(
    "permission",
    [
        Permission.AGENT_READ,
        Permission.MEMORY_READ,
        Permission.RUN_READ,
        Permission.RUN_CANCEL,
    ],
)
def test_read_and_cancel_permissions_cannot_enter_mutation_limiter(
    permission: Permission,
) -> None:
    with pytest.raises(ValueError, match="must use require_permission"):
        rate_limited_permission(permission)


async def test_run_creation_remains_rate_limited(monkeypatch) -> None:
    monkeypatch.setattr(rate_limit_module, "_limiter", RateLimiter(rpm=1))
    dependency = rate_limited_permission(Permission.RUN_CREATE)
    context = _context(Permission.RUN_CREATE)

    assert await dependency(context) == context
    with pytest.raises(HTTPException) as raised:
        await dependency(context)

    assert raised.value.status_code == 429
    assert raised.value.headers is not None
    assert int(raised.value.headers["Retry-After"]) >= 1


async def test_authorized_reads_do_not_consume_mutation_budget(monkeypatch) -> None:
    limiter = RateLimiter(rpm=1)
    monkeypatch.setattr(rate_limit_module, "_limiter", limiter)
    read_dependency = require_permission(Permission.RUN_READ)
    create_dependency = rate_limited_permission(Permission.RUN_CREATE)
    context = _context(Permission.RUN_READ, Permission.RUN_CREATE)

    for _ in range(100):
        assert await read_dependency(context) == context

    assert await create_dependency(context) == context


def test_product_read_routes_use_authorization_without_mutation_limiter() -> None:
    routers = (
        event_memory.router,
        event_setup.router,
        events.router,
        surfaces.router,
        today.router,
    )
    checked = 0
    for router in routers:
        for route in router.routes:
            if "GET" not in (route.methods or set()):
                continue
            checked += 1
            dependency_modules = {
                dependency.call.__module__
                for dependency in route.dependant.dependencies
            }
            assert "aloy_backend.rate_limit" not in dependency_modules, route.path

    assert checked >= 10
