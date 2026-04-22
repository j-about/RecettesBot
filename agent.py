"""Claude Agent SDK integration for recipe extraction.

This module exposes a single coroutine, :func:`extract_recipe`, that takes a
recipe URL and returns a validated :class:`RecipeOutput` describing the recipe
in French. Internally it spawns the Claude Code CLI subprocess via
``claude-agent-sdk``, lets the agent fetch the page with the ``WebFetch`` tool,
asks for a JSON-schema-constrained response, and validates the result against
the Pydantic schema before returning it to the caller.

The function is intentionally side-effect-free with respect to the database:
persistence and embedding are the responsibility of the ``/ajouter`` command
handler, which composes :func:`extract_recipe` with the embedding service and
the SQLModel layer.
"""

import asyncio
from typing import Literal

import logfire
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKError,
    ResultMessage,
    query,
)
from pydantic import BaseModel, Field, ValidationError

from settings import get_settings

__all__ = [
    "RECIPE_SYSTEM_PROMPT",
    "IngredientOutput",
    "RecipeExtractionAgentError",
    "RecipeExtractionContentError",
    "RecipeExtractionError",
    "RecipeExtractionTimeout",
    "RecipeExtractionValidationError",
    "RecipeOutput",
    "extract_recipe",
]


class RecipeExtractionError(Exception):
    """Base class for any failure inside :func:`extract_recipe`.

    The ``/ajouter`` command handler catches the four concrete subclasses
    below and maps each to a French user-facing error message. Catch the
    base class to handle "any extraction failure".
    """


class RecipeExtractionValidationError(RecipeExtractionError):
    """The URL is malformed, or the agent's payload failed schema validation."""


class RecipeExtractionTimeout(RecipeExtractionError):
    """The agent did not return within ``AGENT_TIMEOUT_SECONDS``."""


class RecipeExtractionContentError(RecipeExtractionError):
    """The agent could not extract a recipe from the page content.

    Raised when the agent explicitly reports ``status: "error"`` in its
    structured output (e.g. the site blocked automated access, or the page
    contains no recipe). The exception message carries the agent's
    ``error_message`` so it can be surfaced to the user.
    """


class RecipeExtractionAgentError(RecipeExtractionError):
    """The Claude Agent SDK reported an error or returned no structured output."""


class IngredientOutput(BaseModel):
    """One ingredient line as returned by the agent."""

    name: str
    quantity: float | None = None
    unit: str | None = None


class RecipeOutput(BaseModel):
    """Structured recipe payload returned by the Claude agent.

    The ``status`` field acts as a discriminator: ``"success"`` means the
    recipe was extracted normally; ``"error"`` means the agent could not
    extract a recipe (blocked site, no recipe on page, etc.) and
    ``error_message`` describes the problem.

    On success, ``title``, ``ingredients``, ``steps``, and ``servings``
    mirror the persistence shape of :class:`models.Recipe` minus the
    database-only fields. The ``servings > 0`` constraint matches the
    CHECK on the recipes table so a value that would fail at insert time
    fails earlier, inside :func:`extract_recipe`, with a clean validation
    error.
    """

    status: Literal["success", "error"]
    error_message: str | None = None
    title: str = ""
    ingredients: list[IngredientOutput] = []
    steps: list[str] = []
    servings: int = Field(default=1, gt=0)


# Computed once at import time: Pydantic walks the model tree on every call,
# and the schema is static for the lifetime of the process.
_RECIPE_JSON_SCHEMA = RecipeOutput.model_json_schema()


# System prompt instructing the agent to fetch the page with WebFetch and to
# translate every field (title, ingredients, steps) into French regardless of
# the source language.
RECIPE_SYSTEM_PROMPT = (
    "Tu es un extracteur de recettes de cuisine. Ta mission : récupérer une "
    "page web via l'outil WebFetch, en extraire fidèlement les informations "
    "de la recette, et les retourner en français.\n"
    "\n"
    "## Processus\n"
    "1. Appelle WebFetch avec l'URL fournie par l'utilisateur.\n"
    "2. Identifie la recette principale dans le contenu de la page.\n"
    "3. Extrais le titre, les ingrédients, les étapes et le nombre de "
    "portions.\n"
    "\n"
    "## Règles d'extraction\n"
    "- **status** : « success » si la recette a été extraite avec succès, "
    "« error » si l'extraction a échoué pour quelque raison que ce soit.\n"
    "- **error_message** : null en cas de succès, description du problème "
    "en cas d'erreur.\n"
    "- **title** : titre de la recette, en casse de phrase (majuscule "
    "initiale uniquement). Exemple : « Tarte aux pommes de grand-mère ».\n"
    "- **ingredients** : une entrée par ingrédient. Sépare la quantité "
    "numérique (champ `quantity`, type float), l'unité abrégée (g, ml, cl, "
    "kg, L, c. à s., c. à c.) et le nom de l'ingrédient. Si la quantité "
    "n'est pas précisée (« sel », « poivre »), laisse `quantity` et `unit` "
    "à null. Convertis les fractions en décimales (½ → 0.5).\n"
    "- **steps** : liste ordonnée des étapes. Chaque étape est une phrase "
    "complète et autonome, sans numérotation en début de chaîne.\n"
    "- **servings** : nombre entier de portions tel qu'indiqué dans la "
    "recette. Si absent, déduis-le du contexte ou utilise 4 par défaut.\n"
    "\n"
    "## Traduction\n"
    "Traduis TOUT en français : titre, noms d'ingrédients, étapes. "
    "Exceptions : conserve tels quels les noms propres, marques déposées "
    "et termes culinaires étrangers consacrés (mascarpone, mozzarella, "
    "worcestershire, tofu, etc.).\n"
    "\n"
    "## Fidélité\n"
    "- Extrais uniquement ce qui figure sur la page. N'invente jamais "
    "d'ingrédients, d'étapes ou de quantités absents de la source.\n"
    "- Si la page ne contient pas de recette, ou si le contenu est "
    "inaccessible (site bloqué, erreur réseau, page protégée), retourne "
    "status « error » avec un error_message décrivant brièvement le "
    "problème en français."
)


async def extract_recipe(url: str) -> RecipeOutput:
    """Extract a recipe from ``url`` using the Claude Agent SDK.

    Wrapped in a Logfire span (``agent.extract_recipe``) and a configurable
    ``asyncio.timeout`` (default 120 s via ``AGENT_TIMEOUT_SECONDS``).

    Raises:
        RecipeExtractionValidationError: URL is not http(s), or the agent's
            payload failed Pydantic validation against :class:`RecipeOutput`.
        RecipeExtractionTimeout: agent exceeded ``AGENT_TIMEOUT_SECONDS``.
        RecipeExtractionAgentError: SDK error, agent ``is_error``, or empty
            structured output.
    """
    if not (url.startswith("http://") or url.startswith("https://")):
        raise RecipeExtractionValidationError(f"Invalid URL: {url!r}")

    settings = get_settings()
    prompt = f"Extrais la recette depuis cette URL : {url}"
    options = ClaudeAgentOptions(
        system_prompt=RECIPE_SYSTEM_PROMPT,
        allowed_tools=["WebFetch"],
        permission_mode="acceptEdits",
        model="haiku",
        # Constrain Claude's response to the RecipeOutput shape via the
        # CLI's json_schema mode. The SDK forwards this to the Messages API
        # as the structured-output contract.
        output_format={
            "type": "json_schema",
            "schema": _RECIPE_JSON_SCHEMA,
        },
    )

    with logfire.span("agent.extract_recipe", url=url) as span:
        span.set_attribute("success", False)
        raw = None
        try:
            async with asyncio.timeout(settings.agent_timeout_seconds):
                async for message in query(prompt=prompt, options=options):
                    if isinstance(message, ResultMessage):
                        if message.is_error:
                            raise RecipeExtractionAgentError(
                                message.result or "agent reported error"
                            )
                        if message.structured_output is not None:
                            raw = message.structured_output
                            break
        except TimeoutError as exc:
            raise RecipeExtractionTimeout(
                f"agent timed out after {settings.agent_timeout_seconds}s"
            ) from exc
        except ClaudeSDKError as exc:
            raise RecipeExtractionAgentError(str(exc)) from exc

        if raw is None:
            raise RecipeExtractionAgentError("agent did not return structured output")

        try:
            result = RecipeOutput.model_validate(raw)
        except ValidationError as exc:
            raise RecipeExtractionValidationError("agent output failed schema validation") from exc

        if result.status == "error":
            raise RecipeExtractionContentError(result.error_message or "extraction failed")

        if not result.ingredients:
            raise RecipeExtractionValidationError("recipe has no ingredients")

        span.set_attribute("success", True)
        return result
