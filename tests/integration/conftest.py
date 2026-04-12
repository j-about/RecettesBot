"""Fixtures for DB-backed integration tests.

These tests need a real PostgreSQL + pgvector instance — bring it up with
``docker compose up -d`` first. If the DB is unreachable the autouse
``_database_available`` fixture skips every test in this package cleanly.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine

from db import dispose_engine, get_engine
from settings import get_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = REPO_ROOT / "alembic.ini"


def _async_url_to_raw(url: str) -> str:
    """Convert a SQLAlchemy async URL to the bare DSN that ``asyncpg.connect`` expects."""
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


@pytest.fixture(scope="session", autouse=True)
def _database_available() -> None:
    """Skip every test in this package if PostgreSQL is not reachable."""
    settings = get_settings()
    raw_url = _async_url_to_raw(settings.database_url)

    async def _ping() -> None:
        conn = await asyncpg.connect(raw_url, timeout=2)
        await conn.close()

    try:
        asyncio.run(_ping())
    except Exception as exc:  # noqa: BLE001 — any failure means skip
        pytest.skip(
            f"PostgreSQL not reachable at {settings.database_url}: {exc}. "
            "Run `docker compose up -d` first.",
            allow_module_level=False,
        )


@pytest.fixture(scope="session", autouse=True)
def _apply_migrations(_database_available: None) -> None:
    """Run ``alembic upgrade head`` once per test session.

    The upgrade is idempotent (the only migration so far is
    ``CREATE EXTENSION IF NOT EXISTS vector``), so it can run on a fresh DB or
    one that's already up-to-date. The schema is intentionally left in place
    after the session ends — devs can downgrade explicitly via the CLI if
    they want a clean slate.
    """
    cfg = Config(str(ALEMBIC_INI))
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    """Yield the cached async engine for the duration of one test.

    Function-scoped on purpose: asyncpg connections are bound to the event
    loop they were created on, and pytest-asyncio uses a fresh loop per test
    by default. Disposing + clearing the LRU cache after each test forces the
    next test to rebuild the engine inside its own loop.
    """
    yield get_engine()
    await dispose_engine()
