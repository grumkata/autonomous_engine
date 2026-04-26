"""
Alembic async migration environment.

Setup steps (one-time):
    pip install alembic
    alembic init db/migrations      # creates alembic.ini + this env.py
    # Then replace env.py with this file.

    # Generate your first migration:
    alembic revision --autogenerate -m "initial schema"

    # Run migrations:
    alembic upgrade head

This file configures Alembic to use SQLAlchemy async (required for asyncpg
and aiosqlite). It reads DATABASE_URL from the environment so the same
command works in dev and prod.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import your metadata so Alembic can detect table changes.
from db.tables import Base

# Alembic Config object, parsed from alembic.ini.
config = context.config

# Set up Python logging from alembic.ini [loggers] section.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata for autogenerate.
target_metadata = Base.metadata


def get_url() -> str:
    import os
    url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./engine.db")
    return url


def run_migrations_offline() -> None:
    """
    Run migrations without a live DB connection.
    Generates SQL to stdout — useful for reviewing before applying.
    """
    url = get_url()
    context.configure(
        url=url,
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
    """Run migrations using the async engine."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # use NullPool for migrations (no pooling needed)
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
