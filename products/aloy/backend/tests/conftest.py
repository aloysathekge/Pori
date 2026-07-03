import pytest
import pytest_asyncio
from fastapi import Request
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from pori_cloud.api import app
from pori_cloud.auth import get_current_user
from pori_cloud.database import get_session

TEST_USER_ID = "test-user"


@pytest_asyncio.fixture
async def db_session_maker(tmp_path):
    db_path = tmp_path / "pori_cloud_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    yield session_maker
    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session_maker):
    async def override_get_session():
        async with db_session_maker() as session:
            yield session

    async def override_get_current_user(request: Request):
        return request.headers.get("X-Test-User", TEST_USER_ID)

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_current_user] = override_get_current_user

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
