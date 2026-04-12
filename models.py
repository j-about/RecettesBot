"""SQLModel data definitions for RecettesBot.

``Recipe`` and ``Ingredient`` form a one-to-many relationship. Both classes are
simultaneously SQLAlchemy ORM tables and Pydantic validation schemas,
so the same definition serves persistence, agent output validation, and API
serialisation without a parallel "schema" hierarchy.

Importing this module has the side effect of registering both tables on
``SQLModel.metadata``; ``migrations/env.py`` relies on that for autogenerate.

Note: this module deliberately omits ``from __future__ import annotations``.
SQLModel / SQLAlchemy resolve relationship targets by inspecting the runtime
annotations on the class, and PEP 563's stringification turns
``list["Ingredient"]`` into a doubly-quoted form that SQLAlchemy cannot parse,
producing ``relationship("list['Ingredient']")`` errors at mapper config time.
PEP 604 unions (``int | None``, ``list[float] | None``) work natively on
Python 3.13 without the future import, so nothing else needs adjusting.
"""

from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, CheckConstraint, Column, DateTime, Index, Text, func
from sqlmodel import Field, Relationship, SQLModel

__all__ = ["Ingredient", "Recipe", "adjust_quantity", "format_ingredient_line", "format_quantity"]


def adjust_quantity(original_qty: float | None, n: int, original_servings: int) -> float | None:
    """Scale *original_qty* from *original_servings* to *n* servings.

    Returns ``None`` unchanged when the ingredient has no quantity.
    """
    if original_qty is None:
        return None
    return original_qty * n / original_servings


def format_quantity(value: float) -> str:
    """Format a quantity for display, avoiding scientific notation."""
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0")


def format_ingredient_line(
    name: str,
    quantity: float | None,
    unit: str | None,
    servings: int,
    original_servings: int,
) -> str:
    """Format one ingredient for display (e.g. ``500 g farine``).

    Adjusts *quantity* from *original_servings* to *servings* via
    :func:`adjust_quantity`, then assembles ``"qty unit name"``.
    """
    adjusted = adjust_quantity(quantity, servings, original_servings)
    parts: list[str] = []
    if adjusted is not None:
        parts.append(format_quantity(adjusted))
    if unit:
        parts.append(unit)
    parts.append(name)
    return " ".join(parts)


class Recipe(SQLModel, table=True):
    """A user-saved recipe with a title embedding for semantic search."""

    __tablename__ = "recipes"
    __table_args__ = (
        CheckConstraint("servings > 0", name="ck_recipes_servings_positive"),
        # HNSW index for cosine similarity search on the title embedding.
        # Declared here so ``alembic --autogenerate`` sees
        # it as part of the model and never produces a phantom drop. Parameters
        # m=16 / ef_construction=64 are the pgvector defaults that work well
        # from the first row inserted.
        Index(
            "ix_recipes_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    telegram_user_id: int = Field(sa_type=BigInteger, index=True, nullable=False)
    source_url: str = Field(sa_type=Text, nullable=False)
    title: str = Field(sa_type=Text, nullable=False)
    servings: int = Field(nullable=False)
    steps: str = Field(sa_type=Text, nullable=False)
    embedding: list[float] | None = Field(
        default=None,
        sa_column=Column(Vector(768)),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
        ),
    )

    ingredients: list["Ingredient"] = Relationship(
        back_populates="recipe",
        sa_relationship_kwargs={"lazy": "selectin"},
    )


class Ingredient(SQLModel, table=True):
    """A single ingredient line attached to a :class:`Recipe`."""

    __tablename__ = "ingredients"

    id: int | None = Field(default=None, primary_key=True)
    recipe_id: int = Field(foreign_key="recipes.id", index=True, nullable=False)
    name: str = Field(sa_type=Text, nullable=False)
    quantity: float | None = None
    unit: str | None = Field(default=None, sa_type=Text)

    recipe: Recipe | None = Relationship(back_populates="ingredients")
