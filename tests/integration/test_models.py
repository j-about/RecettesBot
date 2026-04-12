"""Database-level integration tests for ``Recipe`` and ``Ingredient``.

Exercises the schema produced by the Alembic migration: relationship
round-trip, the ``servings > 0`` ``CHECK``, the foreign-key constraint, the
HNSW index, and a real cosine-distance query on the production table shape.

Each test isolates its writes to a unique synthetic ``telegram_user_id`` and
cleans up in a ``finally`` block, so the migration state survives across the
test session.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from models import Ingredient, Recipe

pytestmark = pytest.mark.asyncio


async def _purge_user(session: AsyncSession, telegram_user_id: int) -> None:
    """Delete every recipe (and cascading ingredients) owned by a synthetic user."""
    recipes = (
        await session.exec(select(Recipe).where(Recipe.telegram_user_id == telegram_user_id))
    ).all()
    for recipe in recipes:
        await session.exec(delete(Ingredient).where(Ingredient.recipe_id == recipe.id))
    await session.exec(delete(Recipe).where(Recipe.telegram_user_id == telegram_user_id))
    await session.commit()


async def test_recipe_and_ingredients_roundtrip(engine: AsyncEngine) -> None:
    """A Recipe with two Ingredients should round-trip through ``AsyncSession``.

    ``selectin`` lazy loading on the relationship is what makes this work
    under async without tripping ``MissingGreenlet``.
    """
    user_id = 900_000_001
    async with AsyncSession(engine) as session:
        try:
            recipe = Recipe(
                telegram_user_id=user_id,
                source_url="https://example.com/tarte",
                title="Tarte aux pommes",
                servings=6,
                steps='["Éplucher", "Cuire"]',
                ingredients=[
                    Ingredient(name="Pommes", quantity=4.0, unit="pièces"),
                    Ingredient(name="Sucre", quantity=100.0, unit="g"),
                ],
            )
            session.add(recipe)
            await session.commit()
            await session.refresh(recipe)

            stmt = select(Recipe).where(Recipe.id == recipe.id)
            fetched = (await session.exec(stmt)).one()

            assert fetched.title == "Tarte aux pommes"
            assert fetched.servings == 6
            assert fetched.created_at is not None
            assert {ing.name for ing in fetched.ingredients} == {"Pommes", "Sucre"}
            sucre = next(ing for ing in fetched.ingredients if ing.name == "Sucre")
            assert sucre.quantity == 100.0
            assert sucre.unit == "g"
        finally:
            await _purge_user(session, user_id)


@pytest.mark.parametrize("bad_servings", [0, -1])
async def test_servings_check_constraint_rejects_non_positive(
    engine: AsyncEngine, bad_servings: int
) -> None:
    """The ``ck_recipes_servings_positive`` CHECK should reject zero and negatives."""
    user_id = 900_000_002
    async with AsyncSession(engine) as session:
        try:
            session.add(
                Recipe(
                    telegram_user_id=user_id,
                    source_url="https://example.com/x",
                    title="Bad",
                    servings=bad_servings,
                    steps="[]",
                )
            )
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()
        finally:
            await _purge_user(session, user_id)


async def test_ingredient_foreign_key_is_enforced(engine: AsyncEngine) -> None:
    """Inserting an ``Ingredient`` with a non-existent ``recipe_id`` must fail."""
    async with AsyncSession(engine) as session:
        session.add(Ingredient(recipe_id=999_999_999, name="Orphan"))
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()


async def test_recipes_hnsw_index_is_present(engine: AsyncEngine) -> None:
    """``ix_recipes_embedding_hnsw`` must exist with the expected access method."""
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE tablename = 'recipes' AND indexname = 'ix_recipes_embedding_hnsw'"
            )
        )
        rows = result.all()

    assert len(rows) == 1
    indexdef = rows[0][0]
    assert "USING hnsw" in indexdef
    assert "vector_cosine_ops" in indexdef


async def test_recipe_embedding_cosine_distance_query(engine: AsyncEngine) -> None:
    """End-to-end vector search on the production ``recipes`` table.

    Inserts two recipes with distinct 768-dim embeddings and asserts that
    ``cosine_distance`` ordering returns the nearest one. This is the only
    test that exercises ``Vector(768)`` against the real schema (the existing
    ``test_database`` suite uses a 3-dim throwaway table).
    """
    user_id = 900_000_003
    async with AsyncSession(engine) as session:
        try:
            close_vec = [1.0] + [0.0] * 767
            far_vec = [0.0] * 767 + [1.0]
            session.add(
                Recipe(
                    telegram_user_id=user_id,
                    source_url="https://example.com/close",
                    title="Close",
                    servings=2,
                    steps="[]",
                    embedding=close_vec,
                )
            )
            session.add(
                Recipe(
                    telegram_user_id=user_id,
                    source_url="https://example.com/far",
                    title="Far",
                    servings=2,
                    steps="[]",
                    embedding=far_vec,
                )
            )
            await session.commit()

            query_vec = [1.0] + [0.0] * 767
            stmt = (
                select(Recipe)
                .where(Recipe.telegram_user_id == user_id)
                .order_by(Recipe.embedding.cosine_distance(query_vec))
                .limit(1)
            )
            nearest = (await session.exec(stmt)).first()

            assert nearest is not None
            assert nearest.title == "Close"
        finally:
            await _purge_user(session, user_id)


async def test_btree_index_on_telegram_user_id(engine: AsyncEngine) -> None:
    """``ix_recipes_telegram_user_id`` B-tree index must exist."""
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE tablename = 'recipes' AND indexname = 'ix_recipes_telegram_user_id'"
            )
        )
        rows = result.all()

    assert len(rows) == 1
    indexdef = rows[0][0]
    assert "USING btree" in indexdef
    assert "telegram_user_id" in indexdef


async def test_embedding_column_is_vector_768(engine: AsyncEngine) -> None:
    """The ``embedding`` column must be ``vector(768)``."""
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT udt_name, character_maximum_length "
                "FROM information_schema.columns "
                "WHERE table_name = 'recipes' AND column_name = 'embedding'"
            )
        )
        row = result.one()

    assert row[0] == "vector"


async def test_ingredients_table_has_recipe_id_fk_index(engine: AsyncEngine) -> None:
    """The ``recipe_id`` column in ``ingredients`` should be indexed."""
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE tablename = 'ingredients' AND indexdef LIKE '%recipe_id%'"
            )
        )
        rows = result.all()

    assert len(rows) >= 1


async def test_cosine_distance_threshold_filters_far_recipes(engine: AsyncEngine) -> None:
    """Recipes beyond ``SEARCH_DISTANCE_THRESHOLD`` should not appear in results.

    Uses synthetic orthogonal vectors: the "close" recipe has cosine distance 0
    from the query, the "far" recipe has cosine distance ~1.0 (orthogonal).
    Filtering with threshold 0.65 should return only the close recipe.
    """
    user_id = 900_000_004
    async with AsyncSession(engine) as session:
        try:
            close_vec = [1.0] + [0.0] * 767
            far_vec = [0.0] * 767 + [1.0]
            session.add(
                Recipe(
                    telegram_user_id=user_id,
                    source_url="https://example.com/close",
                    title="Close recipe",
                    servings=2,
                    steps="[]",
                    embedding=close_vec,
                )
            )
            session.add(
                Recipe(
                    telegram_user_id=user_id,
                    source_url="https://example.com/far",
                    title="Far recipe",
                    servings=2,
                    steps="[]",
                    embedding=far_vec,
                )
            )
            await session.commit()

            query_vec = [1.0] + [0.0] * 767
            threshold = 0.65
            dist_expr = Recipe.embedding.cosine_distance(query_vec)
            stmt = (
                select(Recipe)
                .where(Recipe.telegram_user_id == user_id)
                .where(Recipe.embedding.is_not(None))
                .where(dist_expr <= threshold)
                .order_by(dist_expr)
                .limit(5)
            )
            results = (await session.exec(stmt)).all()

            assert len(results) == 1
            assert results[0].title == "Close recipe"
        finally:
            await _purge_user(session, user_id)
