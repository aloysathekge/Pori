"""Async SQLAlchemy/SQLModel database plumbing: the module-level ``engine``
and ``async_session`` factory, the ``get_session`` FastAPI dependency, and
``init_db`` (dev-time ``create_all`` — production schema changes go through
Alembic). ``_async_url`` upgrades sync postgres URLs to asyncpg.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from .config import settings


def _async_url(url: str) -> str:
    """Ensure the database URL uses an async driver."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


engine = create_async_engine(_async_url(settings.database_url), echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    """Create tables on startup (dev convenience). Use Alembic for production."""
    from .models import Run  # noqa: F401 — ensure models are registered

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session():
    """FastAPI dependency that yields an async database session."""
    async with async_session() as session:
        yield session
