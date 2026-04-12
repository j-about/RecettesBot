"""Builder functions for test data: Recipe, Ingredient, and RecipeOutput.

These factories produce valid model instances with sensible defaults for use
in both unit and e2e tests. All fields can be overridden via keyword arguments.
"""

from agent import IngredientOutput, RecipeOutput
from models import Ingredient, Recipe


def sample_recipe_output(
    *,
    title: str = "Tarte aux pommes",
    servings: int = 6,
    ingredients: list[IngredientOutput] | None = None,
    steps: list[str] | None = None,
) -> RecipeOutput:
    """Build a valid ``RecipeOutput`` as returned by the agent."""
    if ingredients is None:
        ingredients = [
            IngredientOutput(name="Pommes", quantity=4.0, unit="pièces"),
            IngredientOutput(name="Sucre", quantity=100.0, unit="g"),
        ]
    if steps is None:
        steps = ["Éplucher", "Cuire"]
    return RecipeOutput(
        status="success",
        title=title,
        ingredients=ingredients,
        steps=steps,
        servings=servings,
    )


def make_fake_recipe(
    *,
    id: int = 1,
    title: str = "Tarte aux pommes",
    servings: int = 6,
    telegram_user_id: int = 42,
    steps: str = "[]",
    embedding: list[float] | None = None,
) -> Recipe:
    """Build a ``Recipe`` instance without a database round-trip."""
    return Recipe(
        id=id,
        telegram_user_id=telegram_user_id,
        source_url="https://example.com",
        title=title,
        servings=servings,
        steps=steps,
        embedding=embedding,
    )


def make_fake_ingredients(
    recipe_id: int = 1,
    *,
    specs: list[tuple[str, float | None, str | None]] | None = None,
) -> list[Ingredient]:
    """Build a list of ``Ingredient`` instances from ``(name, qty, unit)`` tuples.

    When *specs* is ``None`` a default pair of ingredients is returned.
    """
    if specs is None:
        specs = [
            ("Pommes", 4.0, "pièces"),
            ("Sucre", 100.0, "g"),
        ]
    return [
        Ingredient(id=i + 1, recipe_id=recipe_id, name=name, quantity=qty, unit=unit)
        for i, (name, qty, unit) in enumerate(specs)
    ]
