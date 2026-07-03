import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from pori_cloud.config import settings
from pori_cloud.database import _async_url
from pori_cloud.memory_store import MemorySnapshot  # noqa: F401
from pori_cloud.models import (  # noqa: F401
    AgentConfig,
    ContextArtifact,
    Conversation,
    CoreMemoryBlock,
    EvolutionActivation,
    EvolutionProposal,
    KnowledgeEntry,
    Message,
    Organization,
    OrganizationMembership,
    Run,
    SkillDefinition,
    SkillGrant,
    TeamConfig,
    TraceRecord,
    UsageRecord,
    UserProfile,
)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata

db_url = _async_url(settings.database_url)


def run_migrations_offline() -> None:
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = create_async_engine(db_url, poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
