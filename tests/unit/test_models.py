"""Pydantic-level validation tests for the SQLModel data classes.

These tests do not touch a database — they exercise only the validation
contract of ``Recipe`` and ``Ingredient``. Database-level constraints
(``CHECK``, foreign keys, the HNSW index) are covered by
``tests/integration/test_models.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from models import Ingredient, Recipe, adjust_quantity, format_quantity


def _minimal_recipe_kwargs() -> dict[str, object]:
    return {
        "telegram_user_id": 1,
        "source_url": "https://example.com/recette",
        "title": "Tarte aux pommes",
        "servings": 4,
        "steps": '["Éplucher les pommes", "Cuire 30 minutes"]',
    }


def test_recipe_construction_with_minimal_fields() -> None:
    recipe = Recipe(**_minimal_recipe_kwargs())

    assert recipe.id is None
    assert recipe.telegram_user_id == 1
    assert recipe.title == "Tarte aux pommes"
    assert recipe.servings == 4
    assert recipe.embedding is None
    assert isinstance(recipe.created_at, datetime)
    assert recipe.created_at.tzinfo is UTC


def test_recipe_accepts_768_dimensional_embedding() -> None:
    embedding = [float(i) / 768 for i in range(768)]
    recipe = Recipe(**_minimal_recipe_kwargs(), embedding=embedding)

    assert recipe.embedding is not None
    assert recipe.embedding == embedding
    assert len(recipe.embedding) == 768


def test_recipe_model_dump_json_roundtrip() -> None:
    recipe = Recipe(**_minimal_recipe_kwargs(), embedding=[0.1, 0.2, 0.3])

    payload = recipe.model_dump_json()
    rebuilt = Recipe.model_validate_json(payload)

    assert rebuilt.telegram_user_id == recipe.telegram_user_id
    assert rebuilt.title == recipe.title
    assert rebuilt.servings == recipe.servings
    assert rebuilt.steps == recipe.steps
    assert rebuilt.embedding == [0.1, 0.2, 0.3]


def test_ingredient_construction_with_optional_fields_omitted() -> None:
    ingredient = Ingredient(recipe_id=1, name="Sucre")

    assert ingredient.id is None
    assert ingredient.recipe_id == 1
    assert ingredient.name == "Sucre"
    assert ingredient.quantity is None
    assert ingredient.unit is None


def test_ingredient_construction_with_all_fields() -> None:
    ingredient = Ingredient(recipe_id=42, name="Farine", quantity=250.0, unit="g")

    assert ingredient.recipe_id == 42
    assert ingredient.name == "Farine"
    assert ingredient.quantity == 250.0
    assert ingredient.unit == "g"


def test_recipe_and_ingredient_have_relationship_attributes() -> None:
    """Cheap structural check that the bidirectional relationship is wired.

    The actual round-trip through the database is covered in the integration
    suite (``tests/integration/test_models.py``).
    """
    assert hasattr(Recipe, "ingredients")
    assert hasattr(Ingredient, "recipe")


# --- adjust_quantity -------------------------------------------------------


def test_adjust_quantity_doubles() -> None:
    assert adjust_quantity(100.0, 4, 2) == 200.0


def test_adjust_quantity_exact_ratio() -> None:
    assert adjust_quantity(3.0, 6, 3) == 6.0


def test_adjust_quantity_none_passthrough() -> None:
    assert adjust_quantity(None, 4, 2) is None


def test_adjust_quantity_zero_stays_zero() -> None:
    assert adjust_quantity(0.0, 4, 2) == 0.0


def test_adjust_quantity_ratio_one() -> None:
    assert adjust_quantity(250.0, 4, 4) == 250.0


def test_adjust_quantity_large_n() -> None:
    assert adjust_quantity(100.0, 2000, 4) == 50000.0


# --- format_quantity -------------------------------------------------------


def test_format_quantity_integer() -> None:
    assert format_quantity(1250.0) == "1250"


def test_format_quantity_simple_decimal() -> None:
    assert format_quantity(1.5) == "1.5"


def test_format_quantity_two_decimals() -> None:
    assert format_quantity(66.667) == "66.67"


def test_format_quantity_repeating_decimal() -> None:
    assert format_quantity(0.333) == "0.33"


def test_format_quantity_zero() -> None:
    assert format_quantity(0.0) == "0"
