"""Fixtures for end-to-end command-flow tests.

E2e tests exercise complete user journeys through the bot handlers with
all external dependencies (agent SDK, embedding model, database) replaced
by fakes. The ``patch_bot_deps`` autouse fixture wires up everything so
each test starts with a clean, deterministic environment.

The core abstraction is :class:`InMemoryDatabase` — a dict-backed store
that persists recipes across sequential handler calls within a single test,
enabling the full chain ``/ajouter → /chercher → select → /personnes →
/courses → poll answer``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

import bot
from models import Recipe
from tests.mocks.stubs import stub_encode_text, stub_extract_recipe
from tests.mocks.telegram import FakeUpdate


class InMemoryDatabase:
    """Dict-backed recipe store that survives across handler calls."""

    def __init__(self) -> None:
        self.recipes: dict[int, Recipe] = {}
        self._next_id: int = 1

    def add(self, recipe: Recipe) -> None:
        recipe.id = self._next_id
        self._next_id += 1
        for ing in recipe.ingredients or []:
            ing.recipe_id = recipe.id
        self.recipes[recipe.id] = recipe

    def get(self, recipe_id: int) -> Recipe | None:
        return self.recipes.get(recipe_id)

    def search(self, user_id: int) -> list[Recipe]:
        return [r for r in self.recipes.values() if r.telegram_user_id == user_id]


class FakeE2eSession:
    """Combined session supporting add, get, and exec against :class:`InMemoryDatabase`."""

    def __init__(self, db: InMemoryDatabase) -> None:
        self._db = db
        self._pending: list[Any] = []

    def add(self, instance: Any) -> None:
        self._pending.append(instance)

    async def get(self, model: type, pk: Any) -> Any:
        return self._db.get(pk)

    async def exec(self, stmt: Any) -> _FakeResult:
        # Simplified: return all recipes (chercher's WHERE filtering is
        # handled by SQLAlchemy in production; in e2e tests we just return
        # everything and let the handler format the results).
        return _FakeResult(list(self._db.recipes.values()))

    async def commit(self) -> None:
        for obj in self._pending:
            if isinstance(obj, Recipe):
                self._db.add(obj)
        self._pending.clear()

    async def rollback(self) -> None:
        self._pending.clear()


class _FakeResult:
    """Wraps a list to satisfy ``(await session.exec(stmt)).all()``."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def all(self) -> list[Any]:
        return list(self._items)

    def first(self) -> Any:
        return self._items[0] if self._items else None


@pytest.fixture
def fake_db() -> InMemoryDatabase:
    """Fresh in-memory database for each test."""
    return InMemoryDatabase()


@pytest.fixture(autouse=True)
def patch_bot_deps(monkeypatch: pytest.MonkeyPatch, fake_db: InMemoryDatabase) -> None:
    """Wire up all bot dependencies to fakes.

    - ``extract_recipe`` → sample RecipeOutput
    - ``encode_text`` → deterministic 768-dim zero vector
    - ``session_scope`` → in-memory database
    - ``generate_recipe_pdf`` → fake PDF bytes
    - ``Update`` → FakeUpdate (for error handler isinstance check)
    """
    monkeypatch.setattr(bot, "extract_recipe", stub_extract_recipe())
    monkeypatch.setattr(bot, "encode_text", stub_encode_text())
    monkeypatch.setattr(bot, "Update", FakeUpdate)
    monkeypatch.setattr(bot, "generate_recipe_pdf", lambda r, i, s: b"%PDF-e2e-test")

    @asynccontextmanager
    async def _session_scope() -> AsyncIterator[FakeE2eSession]:
        session = FakeE2eSession(fake_db)
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

    monkeypatch.setattr(bot, "session_scope", _session_scope)
