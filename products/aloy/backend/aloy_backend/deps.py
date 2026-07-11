"""Shared route-layer loading helpers.

``load_owned(Model)`` returns an async loader implementing the ubiquitous
"get by id + organization ownership check + 404" pattern that every
tenant-scoped route repeats. The 404 detail is ``"<ModelName> not found"``,
matching the hand-written messages it replaces.
"""

from __future__ import annotations

from typing import Awaitable, Callable, TypeVar

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from .tenancy import OrganizationContext

T = TypeVar("T")


def load_owned(
    model: type[T],
) -> Callable[[str, OrganizationContext, AsyncSession], Awaitable[T]]:
    """Factory: an async loader for ``model`` rows owned by the caller's org.

    The returned coroutine fetches ``model`` by primary key and raises a 404
    unless the row exists AND belongs to ``context.organization_id`` — a
    foreign id is indistinguishable from a missing one (no existence oracle).
    """
    label = getattr(model, "__name__", "Resource")

    async def dep(
        entity_id: str,
        context: OrganizationContext,
        session: AsyncSession,
    ) -> T:
        row = await session.get(model, entity_id)
        if not row or getattr(row, "organization_id", None) != context.organization_id:
            raise HTTPException(status_code=404, detail=f"{label} not found")
        return row

    return dep
