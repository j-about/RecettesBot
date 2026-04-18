"""Async database engine and session factory.

Runtime code uses an async engine on top of asyncpg. Alembic uses the sync
URL + psycopg (v3) from its own ``migrations/env.py`` — keep them separate.

Note on pgvector + asyncpg: ``pgvector.asyncpg.register_vector`` installs a
**binary** codec on the asyncpg connection, but ``pgvector.sqlalchemy.VECTOR``
ships a ``bind_processor`` that always converts values to a **text** string
(e.g. ``'[1,2,3]'``). With pgvector ``0.4.2`` (the version pinned for this
project) the two paths are mutually exclusive: if both are active the binary
codec receives a string from SQLAlchemy and explodes with
``could not convert string to float``. Since every runtime call site goes
through SQLAlchemy / SQLModel, we deliberately do **not** register the
asyncpg codec — the SQLAlchemy bind/result processors handle vector
conversion via the text format end-to-end. Any future raw-asyncpg call site
that bypasses SQLAlchemy must register the codec on its own connection.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from settings import get_settings

__all__ = [
    "dispose_engine",
    "get_engine",
    "get_session_factory",
    "session_scope",
]


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Return the cached async engine, creating it on first call."""
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        echo=False,
        pool_size=5,
        max_overflow=15,
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the cached async session factory bound to the engine."""
    return async_sessionmaker(
        bind=get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
    )


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Yield an :class:`AsyncSession`, committing on success and rolling back on error."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Dispose the cached engine; safe to call from shutdown hooks or test teardown."""
    if get_engine.cache_info().currsize:
        engine = get_engine()
        await engine.dispose()
        get_engine.cache_clear()
        get_session_factory.cache_clear()
