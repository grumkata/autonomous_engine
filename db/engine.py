"""
Database engine and session management.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import Settings
from db.tables import Base

logger = logging.getLogger(__name__)

# Singletons
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine(settings: Settings) -> AsyncEngine:
    """
    Create async engine with correct config for SQLite vs Postgres.
    """

    connect_args = {}

    if settings.database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}

        # ✅ SQLite — NO pooling args
        return create_async_engine(
            settings.database_url,
            echo=settings.debug,
            connect_args=connect_args,
        )

    # ✅ Postgres / others — WITH pooling
    return create_async_engine(
        settings.database_url,
        echo=settings.debug,
        connect_args=connect_args,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )


async def init_db(settings: Settings) -> None:
    global _engine, _session_factory

    if _engine is not None:
        logger.warning("init_db called more than once — ignoring.")
        return

    logger.info(
        "Initializing database engine: %s",
        settings.database_url.split("://")[0],
    )

    _engine = _build_engine(settings)

    _session_factory = async_sessionmaker(
        bind=_engine,
        expire_on_commit=False,
        autoflush=False,
        class_=AsyncSession,
    )

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database ready.")


async def close_db() -> None:
    global _engine, _session_factory

    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database engine closed.")


@asynccontextmanager
async def session_factory() -> AsyncGenerator[AsyncSession, None]:
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with session_factory() as db:
        yield db