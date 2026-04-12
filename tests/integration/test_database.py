"""Integration tests for the database layer.

Exercises the real engine, the real Alembic migration, and the real pgvector
codec end-to-end. Requires ``docker compose up -d`` (the conftest will skip
the whole module otherwise).
"""

from __future__ import annotations

from typing import Any

import pytest
from pgvector.sqlalchemy import Vector
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import Field, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

pytestmark = pytest.mark.asyncio


class _ProbeVectorRow(SQLModel, table=True):
    """Throwaway model used only by tests in this module.

    Lives on ``SQLModel.metadata`` for the duration of pytest collection but
    is never imported by ``migrations/env.py``, so autogenerate won't see it.
    """

    __tablename__ = "_probe_vector_row"

    id: int | None = Field(default=None, primary_key=True)
    embedding: Any = Field(sa_type=Vector(3))


async def test_pgvector_extension_enabled(engine: AsyncEngine) -> None:
    """The Alembic migration should leave the ``vector`` extension installed."""
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        )
        rows = result.all()

    assert [r[0] for r in rows] == ["vector"]


async def test_raw_vector_text_roundtrip(engine: AsyncEngine) -> None:
    """A raw SQL roundtrip should serialize and order vectors correctly via the text format.

    Uses string literals so the asyncpg side never has to know about
    parameter types — Postgres parses the ``'[1,2,3]'`` text format
    natively. This is the path our SQLAlchemy ``Vector`` columns rely on
    (see ``db.py`` for why we don't register the asyncpg binary codec).
    """
    async with engine.begin() as conn:
        await conn.execute(
            text("CREATE TEMP TABLE _raw_probe (  id serial PRIMARY KEY,  embedding vector(3))")
        )
        await conn.execute(text("INSERT INTO _raw_probe (embedding) VALUES ('[1,2,3]')"))
        await conn.execute(text("INSERT INTO _raw_probe (embedding) VALUES ('[7,8,9]')"))

        result = await conn.execute(
            text(
                "SELECT id, embedding FROM _raw_probe "
                "ORDER BY embedding <-> '[1,2,3]'::vector LIMIT 1"
            )
        )
        rows = result.all()

    assert len(rows) == 1
    nearest_id, embedding = rows[0]
    assert nearest_id == 1
    # ``text()`` queries skip the SQLAlchemy result_processor, so asyncpg
    # returns the raw pgvector text format. Parse it to compare numerically.
    parsed = [float(v) for v in embedding.strip("[]").split(",")]
    assert parsed == pytest.approx([1.0, 2.0, 3.0])


async def test_sqlmodel_vector_field_roundtrip(engine: AsyncEngine) -> None:
    """End-to-end check: SQLModel + ``Vector`` field + ``AsyncSession``.

    Creates the probe table, inserts two rows via ``AsyncSession``, runs a
    cosine-distance ORDER BY, and asserts the nearest row matches. Drops
    the table on teardown to keep test runs idempotent.
    """
    async with engine.begin() as conn:
        await conn.run_sync(_ProbeVectorRow.__table__.drop, checkfirst=True)
        await conn.run_sync(_ProbeVectorRow.__table__.create)

    try:
        async with AsyncSession(engine) as session:
            session.add(_ProbeVectorRow(embedding=[1.0, 2.0, 3.0]))
            session.add(_ProbeVectorRow(embedding=[10.0, 20.0, 30.0]))
            await session.commit()

            stmt = (
                select(_ProbeVectorRow)
                .order_by(_ProbeVectorRow.embedding.cosine_distance([1.0, 2.0, 3.0]))
                .limit(1)
            )
            result = await session.exec(stmt)
            nearest = result.first()

        assert nearest is not None
        assert [float(x) for x in nearest.embedding] == pytest.approx([1.0, 2.0, 3.0])
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(_ProbeVectorRow.__table__.drop, checkfirst=True)
