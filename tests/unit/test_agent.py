"""Unit tests for ``agent.extract_recipe``.

Every test stubs ``claude_agent_sdk.query`` via monkeypatch — no Claude CLI
subprocess, no real network. The stub accepts the same kwargs as the real
function (``prompt``, ``options``) and yields a sequence of pre-built
``Message`` objects, which lets us exercise every branch of
``extract_recipe``: URL validation, success, schema validation failure,
missing structured output, ``is_error`` result, an in-flight SDK exception,
and timeout.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any

import pytest
from claude_agent_sdk import ClaudeSDKError
from pydantic import ValidationError

import agent
from agent import (
    IngredientOutput,
    RecipeExtractionAgentError,
    RecipeExtractionContentError,
    RecipeExtractionTimeout,
    RecipeExtractionValidationError,
    RecipeOutput,
    extract_recipe,
)
from settings import get_settings

# --- helpers --------------------------------------------------------------


def _valid_payload() -> dict[str, Any]:
    return {
        "status": "success",
        "title": "Tarte aux pommes",
        "ingredients": [
            {"name": "pommes", "quantity": 4.0, "unit": None},
            {"name": "sucre", "quantity": 100.0, "unit": "g"},
        ],
        "steps": ["Éplucher les pommes", "Cuire 30 minutes"],
        "servings": 4,
    }


@dataclass
class _FakeResultMessage:
    """Stand-in for ``claude_agent_sdk.ResultMessage``.

    The real class is a dataclass; ``extract_recipe`` only checks
    ``isinstance(message, ResultMessage)`` and reads ``is_error``,
    ``result``, and ``structured_output``. We register this fake against
    the real ``ResultMessage`` symbol used in ``agent.py`` (see the
    ``patched_query`` fixture).
    """

    is_error: bool = False
    result: str | None = None
    structured_output: Any = None


@dataclass
class _FakeAssistantMessage:
    """Stand-in for ``AssistantMessage`` — anything that is *not* a
    ``ResultMessage``. ``extract_recipe`` should ignore it."""

    content: list[Any] = field(default_factory=list)


def _make_query_stub(
    messages: Iterable[Any],
    *,
    raise_exc: BaseException | None = None,
    sleep_before: float = 0.0,
):
    """Build an async-generator stub for ``claude_agent_sdk.query``.

    - ``raise_exc``: raised on first iteration step instead of yielding.
    - ``sleep_before``: ``await asyncio.sleep(sleep_before)`` between
      yields, used to drive the timeout test deterministically.
    """

    async def fake_query(*, prompt: str, options: Any) -> AsyncIterator[Any]:
        if raise_exc is not None:
            raise raise_exc
        for message in messages:
            if sleep_before:
                await asyncio.sleep(sleep_before)
            yield message

    return fake_query


@pytest.fixture
def patched_query(monkeypatch: pytest.MonkeyPatch):
    """Return a function that installs a query stub on the agent module.

    Also rebinds ``ResultMessage`` so the stub's ``_FakeResultMessage``
    instances pass the ``isinstance(message, ResultMessage)`` check that
    drives ``extract_recipe``.
    """
    monkeypatch.setattr(agent, "ResultMessage", _FakeResultMessage)

    def _install(stub) -> None:
        monkeypatch.setattr(agent, "query", stub)

    return _install


@pytest.fixture(autouse=True)
def fast_timeout(monkeypatch: pytest.MonkeyPatch):
    """Override the agent timeout to 1 s for fast tests.

    All tests other than the dedicated timeout case use a stub that yields
    immediately, so the value just has to be > 0; the dedicated timeout
    test pairs this with ``sleep_before`` greater than the deadline.
    """
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "1")
    get_settings.cache_clear()
    yield


# --- tests ----------------------------------------------------------------


async def test_rejects_non_http_url(patched_query) -> None:
    # query() should never even be invoked: URL validation runs first.
    patched_query(_make_query_stub([], raise_exc=AssertionError("called")))

    with pytest.raises(RecipeExtractionValidationError, match="Invalid URL"):
        await extract_recipe("javascript:alert(1)")


async def test_returns_validated_output(patched_query) -> None:
    payload = _valid_payload()
    patched_query(_make_query_stub([_FakeResultMessage(structured_output=payload)]))

    result = await extract_recipe("https://example.com/recette")

    assert isinstance(result, RecipeOutput)
    assert result.title == "Tarte aux pommes"
    assert result.servings == 4
    assert len(result.ingredients) == 2
    assert result.ingredients[0].name == "pommes"
    assert result.ingredients[1].unit == "g"
    assert result.steps[0].startswith("Éplucher")


async def test_raises_on_schema_validation_failure(patched_query) -> None:
    bad_payload = _valid_payload()
    del bad_payload["status"]  # required field
    patched_query(_make_query_stub([_FakeResultMessage(structured_output=bad_payload)]))

    with pytest.raises(RecipeExtractionValidationError, match="schema"):
        await extract_recipe("https://example.com/recette")


async def test_raises_when_no_structured_output(patched_query) -> None:
    # Stream contains only assistant messages, never a ResultMessage.
    patched_query(_make_query_stub([_FakeAssistantMessage(), _FakeAssistantMessage()]))

    with pytest.raises(RecipeExtractionAgentError, match="did not return structured output"):
        await extract_recipe("https://example.com/recette")


async def test_raises_when_result_is_error(patched_query) -> None:
    patched_query(_make_query_stub([_FakeResultMessage(is_error=True, result="upstream blew up")]))

    with pytest.raises(RecipeExtractionAgentError, match="upstream blew up"):
        await extract_recipe("https://example.com/recette")


async def test_raises_on_sdk_exception(patched_query) -> None:
    patched_query(_make_query_stub([], raise_exc=ClaudeSDKError("CLI exploded")))

    with pytest.raises(RecipeExtractionAgentError, match="CLI exploded"):
        await extract_recipe("https://example.com/recette")


async def test_times_out(patched_query, monkeypatch: pytest.MonkeyPatch) -> None:
    # Drive the deadline below the stub's per-yield sleep so the
    # asyncio.timeout() context manager fires before the first message
    # arrives. Bypass the autouse fixture's settings via a direct
    # monkeypatch on the imported reference inside agent.
    class _FakeSettings:
        agent_timeout_seconds = 0.05

    monkeypatch.setattr(agent, "get_settings", lambda: _FakeSettings())

    payload = _valid_payload()
    patched_query(
        _make_query_stub(
            [_FakeResultMessage(structured_output=payload)],
            sleep_before=1.0,
        )
    )

    with pytest.raises(RecipeExtractionTimeout, match="timed out"):
        await extract_recipe("https://example.com/recette")


# --- RecipeOutput / IngredientOutput Pydantic validation ------------------


def test_recipe_output_valid_construction() -> None:
    result = RecipeOutput(**_valid_payload())

    assert result.status == "success"
    assert result.title == "Tarte aux pommes"
    assert result.servings == 4
    assert len(result.ingredients) == 2
    assert result.steps == ["Éplucher les pommes", "Cuire 30 minutes"]


def test_recipe_output_rejects_zero_servings() -> None:
    payload = _valid_payload()
    payload["servings"] = 0

    with pytest.raises(ValidationError, match="servings"):
        RecipeOutput(**payload)


def test_recipe_output_error_status_allows_empty_fields() -> None:
    result = RecipeOutput(status="error", error_message="site bloqué")

    assert result.status == "error"
    assert result.error_message == "site bloqué"
    assert result.title == ""
    assert result.ingredients == []
    assert result.steps == []
    assert result.servings == 1


def test_ingredient_output_optional_fields() -> None:
    ing = IngredientOutput(name="Sel")

    assert ing.name == "Sel"
    assert ing.quantity is None
    assert ing.unit is None


# --- extract_recipe: status and content validation -------------------------


async def test_rejects_error_status(patched_query) -> None:
    payload = {
        "status": "error",
        "error_message": "Le site bloque les accès automatisés.",
    }
    patched_query(_make_query_stub([_FakeResultMessage(structured_output=payload)]))

    with pytest.raises(RecipeExtractionContentError, match="bloque les accès"):
        await extract_recipe("https://example.com/recette")


async def test_rejects_error_status_no_message(patched_query) -> None:
    payload = {"status": "error"}
    patched_query(_make_query_stub([_FakeResultMessage(structured_output=payload)]))

    with pytest.raises(RecipeExtractionContentError, match="extraction failed"):
        await extract_recipe("https://example.com/recette")


async def test_rejects_empty_ingredients(patched_query) -> None:
    payload = _valid_payload()
    payload["ingredients"] = []
    patched_query(_make_query_stub([_FakeResultMessage(structured_output=payload)]))

    with pytest.raises(RecipeExtractionValidationError, match="no ingredients"):
        await extract_recipe("https://example.com/recette")
