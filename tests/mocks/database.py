"""Fake database sessions and session-scope replacements.

These fakes satisfy the duck-typed contract that bot handlers use via
``db.session_scope()``. Each variant supports a different subset of
session operations (add, exec, get) so tests can pick the minimal fake.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeSession:
    """Stand-in for ``AsyncSession`` — only ``add`` is exercised."""

    def add(self, instance: object) -> None:
        pass


@dataclass
class FakeSearchResult:
    """Stand-in for the object returned by ``session.exec(stmt)``."""

    _items: list[Any]

    def all(self) -> list[Any]:
        return list(self._items)


@dataclass
class FakeSearchSession:
    """Stand-in for ``AsyncSession`` — supports ``exec`` for search queries."""

    results: list[Any] = field(default_factory=list)

    def add(self, instance: object) -> None:
        pass

    async def exec(self, stmt: Any) -> FakeSearchResult:
        return FakeSearchResult(self.results)


@dataclass
class FakeGetSession:
    """Stand-in for ``AsyncSession`` — supports ``get`` for recipe lookup."""

    recipe: Any = None

    def add(self, instance: object) -> None:
        pass

    async def get(self, model: type, pk: Any) -> Any:
        return self.recipe


@asynccontextmanager
async def fake_session_scope() -> AsyncIterator[FakeSession]:
    """No-op replacement for ``bot.session_scope``."""
    yield FakeSession()


@asynccontextmanager
async def failing_session_scope() -> AsyncIterator[FakeSession]:
    """Replacement for ``bot.session_scope`` that raises on entry."""
    raise RuntimeError("db down")
    yield FakeSession()  # pragma: no cover
