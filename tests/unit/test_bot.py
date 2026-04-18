"""Unit tests for ``bot``.

Exercise the four command handlers and the global error handler directly,
without spawning a python-telegram-bot Application network loop. The fakes
from :mod:`tests.mocks` satisfy the duck-typed attribute contract the
handlers actually read; the ``patch_update`` fixture rebinds ``bot.Update``
to ``FakeUpdate`` so the ``isinstance(update, Update)`` check inside
``bot.error_handler`` accepts our fakes.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import OperationalError as SAOperationalError
from telegram import (
    BotCommand,
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllPrivateChats,
)
from telegram.error import NetworkError as TelegramNetworkError
from telegram.ext import (
    ApplicationHandlerStop,
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    PollAnswerHandler,
    TypeHandler,
)

import bot
from agent import (
    RecipeExtractionAgentError,
    RecipeExtractionContentError,
    RecipeExtractionTimeout,
    RecipeExtractionValidationError,
    RecipeOutput,
)
from embedding import EmbeddingEncodeError, EmbeddingError
from models import Ingredient, Recipe, format_ingredient_line
from pdf import PdfGenerationError
from settings import get_settings
from tests.mocks import (
    FakeBot,
    FakeCallbackQuery,
    FakeChat,
    FakeContext,
    FakeGetSession,
    FakeMessage,
    FakePollAnswer,
    FakeSearchSession,
    FakeUpdate,
    FakeUser,
    failing_session_scope,
    fake_session_scope,
    make_fake_ingredients,
    make_fake_recipe,
    sample_recipe_output,
    stub_encode_text,
    stub_extract_recipe,
)

# --- fixtures -------------------------------------------------------------


@pytest.fixture
def patch_update(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rebind ``bot.Update`` to ``FakeUpdate``.

    Required by tests that exercise ``bot.error_handler``, which uses
    ``isinstance(update, Update)`` to discriminate real updates from
    job-queue errors.
    """
    monkeypatch.setattr(bot, "Update", FakeUpdate)


def _patch_ajouter_deps(
    monkeypatch: pytest.MonkeyPatch,
    *,
    recipe: RecipeOutput | None = None,
    extract_exc: BaseException | None = None,
    encode_exc: BaseException | None = None,
    db_fail: bool = False,
) -> tuple[Any, Any]:
    """Patch all three dependencies used by ``ajouter_command`` after the ACK."""
    extract_stub = stub_extract_recipe(recipe or sample_recipe_output(), raise_exc=extract_exc)
    encode_stub = stub_encode_text(raise_exc=encode_exc)
    monkeypatch.setattr(bot, "extract_recipe", extract_stub)
    monkeypatch.setattr(bot, "encode_text", encode_stub)
    monkeypatch.setattr(
        bot,
        "session_scope",
        failing_session_scope if db_fail else fake_session_scope,
    )
    return extract_stub, encode_stub


# --- /chercher helpers -----------------------------------------------------


def _patch_chercher_deps(
    monkeypatch: pytest.MonkeyPatch,
    *,
    search_results: list[Any] | None = None,
    encode_exc: BaseException | None = None,
    db_fail: bool = False,
) -> Any:
    """Patch ``encode_text`` and ``session_scope`` for ``chercher_command`` tests."""
    encode_stub = stub_encode_text(raise_exc=encode_exc)
    monkeypatch.setattr(bot, "encode_text", encode_stub)

    if db_fail:
        monkeypatch.setattr(bot, "session_scope", failing_session_scope)
    else:
        results = search_results if search_results is not None else []

        @asynccontextmanager
        async def _search_scope() -> AsyncIterator[FakeSearchSession]:
            yield FakeSearchSession(results=results)

        monkeypatch.setattr(bot, "session_scope", _search_scope)

    return encode_stub


def _patch_callback_deps(
    monkeypatch: pytest.MonkeyPatch,
    *,
    recipe: Recipe | None = None,
    pdf_bytes: bytes = b"%PDF-fake",
    pdf_exc: BaseException | None = None,
    db_fail: bool = False,
) -> None:
    """Patch ``session_scope`` and ``generate_recipe_pdf`` for callback tests."""
    if recipe is not None:
        recipe.ingredients = make_fake_ingredients(recipe.id)  # type: ignore[assignment]

    if db_fail:
        monkeypatch.setattr(bot, "session_scope", failing_session_scope)
    else:

        @asynccontextmanager
        async def _get_scope() -> AsyncIterator[FakeGetSession]:
            yield FakeGetSession(recipe=recipe)

        monkeypatch.setattr(bot, "session_scope", _get_scope)

    def _fake_generate(r: Any, ingredients: Any, servings: Any) -> bytes:
        if pdf_exc is not None:
            raise pdf_exc
        return pdf_bytes

    monkeypatch.setattr(bot, "generate_recipe_pdf", _fake_generate)


# --- /ajouter -------------------------------------------------------------


async def test_ajouter_missing_url_prompts() -> None:
    update = FakeUpdate()
    context = FakeContext(args=[])

    result = await bot.ajouter_command(update, context)

    assert result == bot.STATE_AWAITING_URL
    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_AJOUTER_PROMPT, {})]


async def test_ajouter_invalid_url_scheme() -> None:
    update = FakeUpdate()
    context = FakeContext(args=["javascript:alert(1)"])

    await bot.ajouter_command(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_AJOUTER_INVALID_URL, {})]


async def test_ajouter_valid_https_url_acks(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ajouter_deps(monkeypatch)
    update = FakeUpdate()
    context = FakeContext(args=["https://example.com/recette"])

    await bot.ajouter_command(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies[0] == (bot.MSG_AJOUTER_ACK, {})


async def test_ajouter_valid_http_url_acks(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ajouter_deps(monkeypatch)
    update = FakeUpdate()
    context = FakeContext(args=["http://example.com/recette"])

    await bot.ajouter_command(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies[0] == (bot.MSG_AJOUTER_ACK, {})


async def test_ajouter_success_full_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = sample_recipe_output()
    _patch_ajouter_deps(monkeypatch, recipe=recipe)
    update = FakeUpdate()
    context = FakeContext(args=["https://example.com/tarte"])

    await bot.ajouter_command(update, context)

    assert update.effective_message is not None
    replies = [text for text, _ in update.effective_message.replies]
    assert replies == [
        bot.MSG_AJOUTER_ACK,
        bot.MSG_AJOUTER_SUCCESS.format(title="Tarte aux pommes", servings=6),
    ]


async def test_ajouter_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ajouter_deps(
        monkeypatch,
        extract_exc=RecipeExtractionValidationError("bad output"),
    )
    update = FakeUpdate()
    context = FakeContext(args=["https://example.com/not-a-recipe"])

    await bot.ajouter_command(update, context)

    assert update.effective_message is not None
    replies = [text for text, _ in update.effective_message.replies]
    assert replies == [bot.MSG_AJOUTER_ACK, bot.MSG_AJOUTER_VALIDATION_ERROR]


async def test_ajouter_content_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ajouter_deps(
        monkeypatch,
        extract_exc=RecipeExtractionContentError("le site bloque les accès automatisés"),
    )
    update = FakeUpdate()
    context = FakeContext(args=["https://example.com/blocked"])

    await bot.ajouter_command(update, context)

    assert update.effective_message is not None
    replies = [text for text, _ in update.effective_message.replies]
    expected_msg = bot.MSG_AJOUTER_CONTENT_ERROR.format(
        reason="le site bloque les accès automatisés"
    )
    assert replies == [bot.MSG_AJOUTER_ACK, expected_msg]


async def test_ajouter_agent_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ajouter_deps(
        monkeypatch,
        extract_exc=RecipeExtractionAgentError("agent failed"),
    )
    update = FakeUpdate()
    context = FakeContext(args=["https://example.com/recette"])

    await bot.ajouter_command(update, context)

    assert update.effective_message is not None
    replies = [text for text, _ in update.effective_message.replies]
    assert replies == [bot.MSG_AJOUTER_ACK, bot.MSG_AJOUTER_AGENT_ERROR]


async def test_ajouter_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ajouter_deps(
        monkeypatch,
        extract_exc=RecipeExtractionTimeout("timed out"),
    )
    update = FakeUpdate()
    context = FakeContext(args=["https://example.com/recette"])

    await bot.ajouter_command(update, context)

    assert update.effective_message is not None
    replies = [text for text, _ in update.effective_message.replies]
    assert replies == [bot.MSG_AJOUTER_ACK, bot.MSG_AJOUTER_AGENT_ERROR]


async def test_ajouter_embedding_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ajouter_deps(
        monkeypatch,
        encode_exc=EmbeddingEncodeError("encode failed"),
    )
    update = FakeUpdate()
    context = FakeContext(args=["https://example.com/recette"])

    await bot.ajouter_command(update, context)

    assert update.effective_message is not None
    replies = [text for text, _ in update.effective_message.replies]
    assert replies == [bot.MSG_AJOUTER_ACK, bot.MSG_AJOUTER_DB_ERROR]


async def test_ajouter_db_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ajouter_deps(monkeypatch, db_fail=True)
    update = FakeUpdate()
    context = FakeContext(args=["https://example.com/recette"])

    await bot.ajouter_command(update, context)

    assert update.effective_message is not None
    replies = [text for text, _ in update.effective_message.replies]
    assert replies == [bot.MSG_AJOUTER_ACK, bot.MSG_AJOUTER_DB_ERROR]


async def test_ajouter_passes_url_to_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    extract_stub, _ = _patch_ajouter_deps(monkeypatch)
    update = FakeUpdate()
    context = FakeContext(args=["https://example.com/my-recipe"])

    await bot.ajouter_command(update, context)

    assert extract_stub.calls == ["https://example.com/my-recipe"]


async def test_ajouter_passes_title_to_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    _, encode_stub = _patch_ajouter_deps(monkeypatch)
    update = FakeUpdate()
    context = FakeContext(args=["https://example.com/recette"])

    await bot.ajouter_command(update, context)

    assert encode_stub.calls == ["Tarte aux pommes"]


# --- /chercher ------------------------------------------------------------


async def test_chercher_missing_query_prompts() -> None:
    update = FakeUpdate()
    context = FakeContext(args=[])

    result = await bot.chercher_command(update, context)

    assert result == bot.STATE_AWAITING_QUERY
    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_CHERCHER_PROMPT, {})]


async def test_chercher_joins_multi_word_query(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_chercher_deps(monkeypatch, search_results=[])
    update = FakeUpdate()
    context = FakeContext(args=["tarte", "aux", "pommes"])

    await bot.chercher_command(update, context)

    assert update.effective_message is not None
    texts = [text for text, _ in update.effective_message.replies]
    assert texts[0] == bot.MSG_CHERCHER_ACK


async def test_chercher_valid_query_returns_results(monkeypatch: pytest.MonkeyPatch) -> None:
    recipes = [
        make_fake_recipe(id=1, title="Tarte aux pommes", servings=6),
        make_fake_recipe(id=2, title="Compote de pommes", servings=4),
    ]
    _patch_chercher_deps(monkeypatch, search_results=recipes)
    update = FakeUpdate()
    context = FakeContext(args=["pommes"])

    await bot.chercher_command(update, context)

    assert update.effective_message is not None
    replies = update.effective_message.replies
    # ACK + results
    assert len(replies) == 2
    result_text, result_kwargs = replies[1]
    assert result_text == "📋 Résultats :"
    markup = result_kwargs["reply_markup"]
    assert len(markup.inline_keyboard) == 2


async def test_chercher_inline_keyboard_buttons(monkeypatch: pytest.MonkeyPatch) -> None:
    recipes = [make_fake_recipe(id=7, title="Crêpes", servings=4)]
    _patch_chercher_deps(monkeypatch, search_results=recipes)
    update = FakeUpdate()
    context = FakeContext(args=["crêpes"])

    await bot.chercher_command(update, context)

    assert update.effective_message is not None
    _, kwargs = update.effective_message.replies[1]
    markup = kwargs["reply_markup"]
    btn = markup.inline_keyboard[0][0]
    assert btn.text == "Crêpes (4 pers.)"
    assert btn.callback_data == "7"


async def test_chercher_no_results_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_chercher_deps(monkeypatch, search_results=[])
    update = FakeUpdate()
    context = FakeContext(args=["sushi"])

    await bot.chercher_command(update, context)

    assert update.effective_message is not None
    texts = [text for text, _ in update.effective_message.replies]
    assert texts == [
        bot.MSG_CHERCHER_ACK,
        bot.MSG_CHERCHER_NO_RESULTS.format(query="sushi"),
    ]


async def test_chercher_embedding_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_chercher_deps(monkeypatch, encode_exc=EmbeddingEncodeError("boom"))
    update = FakeUpdate()
    context = FakeContext(args=["tarte"])

    await bot.chercher_command(update, context)

    assert update.effective_message is not None
    texts = [text for text, _ in update.effective_message.replies]
    assert texts == [bot.MSG_CHERCHER_ACK, bot.MSG_CHERCHER_EMBEDDING_ERROR]


async def test_chercher_db_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_chercher_deps(monkeypatch, db_fail=True)
    update = FakeUpdate()
    context = FakeContext(args=["tarte"])

    await bot.chercher_command(update, context)

    assert update.effective_message is not None
    texts = [text for text, _ in update.effective_message.replies]
    assert texts == [bot.MSG_CHERCHER_ACK, bot.MSG_CHERCHER_DB_ERROR]


async def test_chercher_passes_query_to_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    encode_stub = _patch_chercher_deps(monkeypatch, search_results=[])
    update = FakeUpdate()
    context = FakeContext(args=["tarte", "pommes"])

    await bot.chercher_command(update, context)

    assert encode_stub.calls == ["tarte pommes"]


# --- callback handler -----------------------------------------------------


async def test_recipe_selected_callback_stores_id(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = make_fake_recipe(id=42)
    _patch_callback_deps(monkeypatch, recipe=recipe)

    msg = FakeMessage()
    query = FakeCallbackQuery(data="42", message=msg)
    update = FakeUpdate(callback_query=query)
    context = FakeContext()

    await bot.recipe_selected_callback(update, context)

    assert context.user_data is not None
    assert context.user_data[bot.USER_DATA_SELECTED_RECIPE_ID] == 42


async def test_recipe_selected_callback_answers_query(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = make_fake_recipe(id=1)
    _patch_callback_deps(monkeypatch, recipe=recipe)

    query = FakeCallbackQuery(data="1")
    update = FakeUpdate(callback_query=query)
    context = FakeContext()

    await bot.recipe_selected_callback(update, context)

    assert query.answered


async def test_recipe_selected_callback_sends_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = make_fake_recipe(id=1, title="Tarte aux pommes")
    _patch_callback_deps(monkeypatch, recipe=recipe, pdf_bytes=b"%PDF-test")

    msg = FakeMessage()
    query = FakeCallbackQuery(data="1", message=msg)
    update = FakeUpdate(callback_query=query)
    context = FakeContext()

    await bot.recipe_selected_callback(update, context)

    assert len(msg.documents) == 1
    assert msg.documents[0]["document"] == b"%PDF-test"
    assert msg.documents[0]["filename"] == "Tarte aux pommes.pdf"


async def test_recipe_selected_callback_uses_adjusted_servings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = make_fake_recipe(id=1, servings=4)
    received_servings: list[int] = []

    def _capture_generate(r: Any, ingredients: Any, servings: Any) -> bytes:
        received_servings.append(servings)
        return b"%PDF-ok"

    _patch_callback_deps(monkeypatch, recipe=recipe)
    monkeypatch.setattr(bot, "generate_recipe_pdf", _capture_generate)

    msg = FakeMessage()
    query = FakeCallbackQuery(data="1", message=msg)
    update = FakeUpdate(callback_query=query)
    context = FakeContext(user_data={bot.USER_DATA_ADJUSTED_SERVINGS: 8})

    await bot.recipe_selected_callback(update, context)

    assert received_servings == [8]


async def test_recipe_selected_callback_clears_adjusted_servings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = make_fake_recipe(id=1)
    _patch_callback_deps(monkeypatch, recipe=recipe)

    msg = FakeMessage()
    query = FakeCallbackQuery(data="1", message=msg)
    update = FakeUpdate(callback_query=query)
    context = FakeContext(user_data={bot.USER_DATA_ADJUSTED_SERVINGS: 10})

    await bot.recipe_selected_callback(update, context)

    assert context.user_data is not None
    assert bot.USER_DATA_ADJUSTED_SERVINGS not in context.user_data


async def test_recipe_selected_callback_recipe_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_callback_deps(monkeypatch, recipe=None)

    msg = FakeMessage()
    query = FakeCallbackQuery(data="999", message=msg)
    update = FakeUpdate(callback_query=query)
    context = FakeContext()

    await bot.recipe_selected_callback(update, context)

    assert msg.replies == [(bot.MSG_RECIPE_NOT_FOUND, {})]
    assert msg.documents == []


async def test_recipe_selected_callback_pdf_error(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = make_fake_recipe(id=1)
    _patch_callback_deps(monkeypatch, recipe=recipe, pdf_exc=PdfGenerationError("render failed"))

    msg = FakeMessage()
    query = FakeCallbackQuery(data="1", message=msg)
    update = FakeUpdate(callback_query=query)
    context = FakeContext()

    await bot.recipe_selected_callback(update, context)

    assert msg.replies == [(bot.MSG_RECIPE_PDF_ERROR, {})]
    assert msg.documents == []


async def test_recipe_selected_callback_send_error(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = make_fake_recipe(id=1)
    _patch_callback_deps(monkeypatch, recipe=recipe)

    msg = FakeMessage(raise_on_document=ConnectionError("network down"))
    query = FakeCallbackQuery(data="1", message=msg)
    update = FakeUpdate(callback_query=query)
    context = FakeContext()

    await bot.recipe_selected_callback(update, context)

    assert msg.replies == [(bot.MSG_RECIPE_SEND_ERROR, {})]


async def test_recipe_selected_callback_invalid_data() -> None:
    query = FakeCallbackQuery(data="not-a-number")
    update = FakeUpdate(callback_query=query)
    context = FakeContext()

    # Must not raise.
    await bot.recipe_selected_callback(update, context)

    assert context.user_data == {}


async def test_recipe_selected_callback_none_query() -> None:
    update = FakeUpdate(callback_query=None)
    context = FakeContext()

    # Must not raise.
    await bot.recipe_selected_callback(update, context)


# --- /personnes -----------------------------------------------------------


async def test_personnes_missing_arg_prompts() -> None:
    update = FakeUpdate()
    context = FakeContext(args=[])

    result = await bot.personnes_command(update, context)

    assert result == bot.STATE_AWAITING_SERVINGS
    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_PERSONNES_PROMPT, {})]


async def test_personnes_non_integer() -> None:
    update = FakeUpdate()
    context = FakeContext(args=["six"])

    await bot.personnes_command(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_PERSONNES_INVALID, {})]


async def test_personnes_zero_rejected() -> None:
    update = FakeUpdate()
    context = FakeContext(args=["0"])

    await bot.personnes_command(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_PERSONNES_INVALID, {})]


async def test_personnes_negative_rejected() -> None:
    update = FakeUpdate()
    context = FakeContext(args=["-3"])

    await bot.personnes_command(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_PERSONNES_INVALID, {})]


async def test_personnes_no_selected_recipe() -> None:
    update = FakeUpdate()
    context = FakeContext(args=["6"], user_data={})

    await bot.personnes_command(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_PERSONNES_NO_RECIPE, {})]


async def test_personnes_stores_adjusted_servings_and_sends_pdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = make_fake_recipe(id=123, servings=4)
    recipe.ingredients = make_fake_ingredients(123)  # type: ignore[assignment]

    @asynccontextmanager
    async def _scope() -> AsyncIterator[FakeGetSession]:
        yield FakeGetSession(recipe=recipe)

    monkeypatch.setattr(bot, "session_scope", _scope)
    monkeypatch.setattr(bot, "generate_recipe_pdf", lambda r, i, s: b"%PDF-personnes")

    update = FakeUpdate()
    context = FakeContext(
        args=["6"],
        user_data={bot.USER_DATA_SELECTED_RECIPE_ID: 123},
    )

    await bot.personnes_command(update, context)

    assert context.user_data is not None
    assert context.user_data[bot.USER_DATA_ADJUSTED_SERVINGS] == 6
    assert update.effective_message is not None
    expected = bot.MSG_PERSONNES_ACK.format(n=6, original_servings=4)
    assert update.effective_message.replies == [(expected, {})]
    assert len(update.effective_message.documents) == 1
    assert update.effective_message.documents[0]["document"] == b"%PDF-personnes"


async def test_personnes_recipe_deleted(monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def _scope() -> AsyncIterator[FakeGetSession]:
        yield FakeGetSession(recipe=None)

    monkeypatch.setattr(bot, "session_scope", _scope)

    update = FakeUpdate()
    context = FakeContext(
        args=["6"],
        user_data={bot.USER_DATA_SELECTED_RECIPE_ID: 999},
    )

    await bot.personnes_command(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_RECIPE_NOT_FOUND, {})]
    assert context.user_data is not None
    assert bot.USER_DATA_ADJUSTED_SERVINGS not in context.user_data


async def test_personnes_large_n_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = make_fake_recipe(id=1, servings=4)
    recipe.ingredients = make_fake_ingredients(1)  # type: ignore[assignment]

    @asynccontextmanager
    async def _scope() -> AsyncIterator[FakeGetSession]:
        yield FakeGetSession(recipe=recipe)

    monkeypatch.setattr(bot, "session_scope", _scope)
    monkeypatch.setattr(bot, "generate_recipe_pdf", lambda r, i, s: b"%PDF-ok")

    update = FakeUpdate()
    context = FakeContext(
        args=["5000"],
        user_data={bot.USER_DATA_SELECTED_RECIPE_ID: 1},
    )

    await bot.personnes_command(update, context)

    assert context.user_data is not None
    assert context.user_data[bot.USER_DATA_ADJUSTED_SERVINGS] == 5000
    assert update.effective_message is not None
    expected = bot.MSG_PERSONNES_ACK.format(n=5000, original_servings=4)
    assert update.effective_message.replies == [(expected, {})]


# --- /courses helpers ------------------------------------------------------


def _make_ingredients(n: int, recipe_id: int = 1) -> list[Ingredient]:
    """Build *n* Ingredient instances with distinct names."""
    return [
        Ingredient(
            id=i + 1,
            recipe_id=recipe_id,
            name=f"Ingrédient {i + 1}",
            quantity=float(i + 1) * 10,
            unit="g",
        )
        for i in range(n)
    ]


def _patch_courses_deps(
    monkeypatch: pytest.MonkeyPatch,
    *,
    recipe: Recipe | None = None,
    db_fail: bool = False,
) -> None:
    """Patch ``session_scope`` for ``courses_command`` tests."""
    if db_fail:
        monkeypatch.setattr(bot, "session_scope", failing_session_scope)
    else:

        @asynccontextmanager
        async def _scope() -> AsyncIterator[FakeGetSession]:
            yield FakeGetSession(recipe=recipe)

        monkeypatch.setattr(bot, "session_scope", _scope)


# --- /courses command -----------------------------------------------------


async def test_courses_no_selected_recipe() -> None:
    update = FakeUpdate()
    context = FakeContext(user_data={})

    await bot.courses_command(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_COURSES_NO_RECIPE, {})]


async def test_courses_recipe_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_courses_deps(monkeypatch, recipe=None)
    update = FakeUpdate()
    context = FakeContext(user_data={bot.USER_DATA_SELECTED_RECIPE_ID: 999})

    await bot.courses_command(update, context)

    assert update.effective_message is not None
    texts = [t for t, _ in update.effective_message.replies]
    assert texts == [bot.MSG_COURSES_ACK, bot.MSG_RECIPE_NOT_FOUND]


async def test_courses_no_ingredients(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = make_fake_recipe(id=1)
    recipe.ingredients = []  # type: ignore[assignment]
    _patch_courses_deps(monkeypatch, recipe=recipe)

    update = FakeUpdate()
    context = FakeContext(user_data={bot.USER_DATA_SELECTED_RECIPE_ID: 1})

    await bot.courses_command(update, context)

    assert update.effective_message is not None
    texts = [t for t, _ in update.effective_message.replies]
    assert texts == [bot.MSG_COURSES_ACK, bot.MSG_COURSES_NO_INGREDIENTS]


async def test_courses_db_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_courses_deps(monkeypatch, db_fail=True)
    update = FakeUpdate()
    context = FakeContext(user_data={bot.USER_DATA_SELECTED_RECIPE_ID: 1})

    await bot.courses_command(update, context)

    assert update.effective_message is not None
    texts = [t for t, _ in update.effective_message.replies]
    assert texts == [bot.MSG_COURSES_ACK, bot.MSG_COURSES_DB_ERROR]


async def test_courses_single_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = make_fake_recipe(id=1, servings=4)
    recipe.ingredients = _make_ingredients(3, recipe_id=1)  # type: ignore[assignment]
    _patch_courses_deps(monkeypatch, recipe=recipe)

    update = FakeUpdate()
    context = FakeContext(user_data={bot.USER_DATA_SELECTED_RECIPE_ID: 1})

    await bot.courses_command(update, context)

    assert update.effective_message is not None
    assert len(update.effective_message.polls) == 1
    poll = update.effective_message.polls[0]
    assert poll["question"] == bot.MSG_COURSES_POLL_QUESTION
    assert poll["options"] == ["Ingrédient 1", "Ingrédient 2", "Ingrédient 3"]
    assert poll["allows_multiple_answers"] is True
    assert poll["is_anonymous"] is False


async def test_courses_multi_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = make_fake_recipe(id=1, servings=4)
    recipe.ingredients = _make_ingredients(15, recipe_id=1)  # type: ignore[assignment]
    _patch_courses_deps(monkeypatch, recipe=recipe)

    update = FakeUpdate()
    context = FakeContext(user_data={bot.USER_DATA_SELECTED_RECIPE_ID: 1})

    await bot.courses_command(update, context)

    assert update.effective_message is not None
    assert len(update.effective_message.polls) == 2
    q1 = update.effective_message.polls[0]["question"]
    q2 = update.effective_message.polls[1]["question"]
    assert q1 == bot.MSG_COURSES_POLL_QUESTION_MULTI.format(start=1, end=10)
    assert q2 == bot.MSG_COURSES_POLL_QUESTION_MULTI.format(start=11, end=15)
    assert len(update.effective_message.polls[0]["options"]) == 10
    assert len(update.effective_message.polls[1]["options"]) == 5


async def test_courses_stores_poll_state(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = make_fake_recipe(id=1, servings=4)
    recipe.ingredients = _make_ingredients(3, recipe_id=1)  # type: ignore[assignment]
    _patch_courses_deps(monkeypatch, recipe=recipe)

    update = FakeUpdate()
    context = FakeContext(user_data={bot.USER_DATA_SELECTED_RECIPE_ID: 1})

    await bot.courses_command(update, context)

    assert "polls" in context.bot_data
    assert len(context.bot_data["polls"]) == 1  # one poll_id entry
    group = next(iter(context.bot_data["polls"].values()))
    assert group["servings"] == 4
    assert group["original_servings"] == 4
    assert len(group["ingredients"]) == 3
    assert group["done"] is False


async def test_courses_with_adjusted_servings(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = make_fake_recipe(id=1, servings=4)
    recipe.ingredients = _make_ingredients(2, recipe_id=1)  # type: ignore[assignment]
    _patch_courses_deps(monkeypatch, recipe=recipe)

    update = FakeUpdate()
    context = FakeContext(
        user_data={
            bot.USER_DATA_SELECTED_RECIPE_ID: 1,
            bot.USER_DATA_ADJUSTED_SERVINGS: 8,
        }
    )

    await bot.courses_command(update, context)

    group = next(iter(context.bot_data["polls"].values()))
    assert group["servings"] == 8
    assert group["original_servings"] == 4


# --- poll_answer_handler --------------------------------------------------


def _setup_poll_group(
    context: FakeContext,
    *,
    ingredients: list[Ingredient] | None = None,
    servings: int = 4,
    original_servings: int = 4,
    poll_count: int = 1,
) -> dict[str, Any]:
    """Create a poll group in *context.bot_data* and return it."""
    if ingredients is None:
        ingredients = make_fake_ingredients()

    chunk_size = 10
    poll_ids = [f"test_poll_{i}" for i in range(poll_count)]
    option_maps: dict[str, dict[int, int]] = {}
    offset = 0
    for _i, pid in enumerate(poll_ids):
        chunk_len = min(chunk_size, len(ingredients) - offset)
        option_maps[pid] = {j: offset + j for j in range(chunk_len)}
        offset += chunk_len

    group: dict[str, Any] = {
        "chat_id": 42,
        "ingredients": ingredients,
        "servings": servings,
        "original_servings": original_servings,
        "poll_ids": poll_ids,
        "option_maps": option_maps,
        "selected_indices": set(),
        "answered_polls": set(),
        "timeout_task": None,
        "done": False,
    }
    context.bot_data["polls"] = {pid: group for pid in poll_ids}
    return group


async def test_poll_answer_single_poll_shopping_list() -> None:
    context = FakeContext()
    group = _setup_poll_group(context)
    poll_id = group["poll_ids"][0]

    update = FakeUpdate(poll_answer=FakePollAnswer(poll_id=poll_id, option_ids=[0, 1]))

    await bot.poll_answer_handler(update, context)

    assert group["done"] is True
    assert len(context.bot.sent_messages) == 1
    text = context.bot.sent_messages[0]["text"]
    assert text.startswith(bot.MSG_COURSES_HEADER)
    assert "Pommes" in text
    assert "Sucre" in text


async def test_poll_answer_multi_poll_completes_on_last() -> None:
    ingredients = _make_ingredients(15)
    context = FakeContext()
    group = _setup_poll_group(context, ingredients=ingredients, poll_count=2)

    # Answer first poll — should not send yet.
    update1 = FakeUpdate(poll_answer=FakePollAnswer(poll_id=group["poll_ids"][0], option_ids=[0]))
    await bot.poll_answer_handler(update1, context)
    assert group["done"] is False
    assert len(context.bot.sent_messages) == 0

    # Answer second poll — now it should send.
    update2 = FakeUpdate(poll_answer=FakePollAnswer(poll_id=group["poll_ids"][1], option_ids=[0]))
    await bot.poll_answer_handler(update2, context)
    assert group["done"] is True
    assert len(context.bot.sent_messages) == 1
    text = context.bot.sent_messages[0]["text"]
    assert "Ingrédient 1" in text
    assert "Ingrédient 11" in text


async def test_poll_answer_empty_selection() -> None:
    context = FakeContext()
    group = _setup_poll_group(context)
    poll_id = group["poll_ids"][0]

    # Answer with no options selected (user retracted vote).
    update = FakeUpdate(poll_answer=FakePollAnswer(poll_id=poll_id, option_ids=[]))
    await bot.poll_answer_handler(update, context)

    assert group["done"] is True
    text = context.bot.sent_messages[0]["text"]
    assert text == bot.MSG_COURSES_EMPTY


async def test_poll_answer_unknown_poll_id() -> None:
    context = FakeContext()
    update = FakeUpdate(poll_answer=FakePollAnswer(poll_id="unknown_poll", option_ids=[0]))

    await bot.poll_answer_handler(update, context)

    assert len(context.bot.sent_messages) == 0


async def test_poll_answer_already_done() -> None:
    context = FakeContext()
    group = _setup_poll_group(context)
    group["done"] = True  # already completed

    update = FakeUpdate(poll_answer=FakePollAnswer(poll_id=group["poll_ids"][0], option_ids=[0]))
    await bot.poll_answer_handler(update, context)

    assert len(context.bot.sent_messages) == 0


async def test_poll_answer_formats_qty_unit_name() -> None:
    ingredients = [
        Ingredient(id=1, recipe_id=1, name="Farine", quantity=500.0, unit="g"),
    ]
    context = FakeContext()
    _setup_poll_group(context, ingredients=ingredients, servings=4, original_servings=4)
    poll_id = list(context.bot_data["polls"])[0]

    update = FakeUpdate(poll_answer=FakePollAnswer(poll_id=poll_id, option_ids=[0]))
    await bot.poll_answer_handler(update, context)

    text = context.bot.sent_messages[0]["text"]
    assert "- 500 g Farine" in text


async def test_poll_answer_none_quantity() -> None:
    ingredients = [
        Ingredient(id=1, recipe_id=1, name="Sel", quantity=None, unit=None),
    ]
    context = FakeContext()
    _setup_poll_group(context, ingredients=ingredients)
    poll_id = list(context.bot_data["polls"])[0]

    update = FakeUpdate(poll_answer=FakePollAnswer(poll_id=poll_id, option_ids=[0]))
    await bot.poll_answer_handler(update, context)

    text = context.bot.sent_messages[0]["text"]
    assert "- Sel" in text
    # No quantity or unit prefix.
    assert "- Sel" in text.split("\n")[-1]


async def test_poll_answer_none_unit() -> None:
    ingredients = [
        Ingredient(id=1, recipe_id=1, name="Oeufs", quantity=3.0, unit=None),
    ]
    context = FakeContext()
    _setup_poll_group(context, ingredients=ingredients, servings=6, original_servings=6)
    poll_id = list(context.bot_data["polls"])[0]

    update = FakeUpdate(poll_answer=FakePollAnswer(poll_id=poll_id, option_ids=[0]))
    await bot.poll_answer_handler(update, context)

    text = context.bot.sent_messages[0]["text"]
    assert "- 3 Oeufs" in text


async def test_poll_answer_with_adjusted_servings() -> None:
    ingredients = [
        Ingredient(id=1, recipe_id=1, name="Farine", quantity=250.0, unit="g"),
    ]
    context = FakeContext()
    _setup_poll_group(context, ingredients=ingredients, servings=8, original_servings=4)
    poll_id = list(context.bot_data["polls"])[0]

    update = FakeUpdate(poll_answer=FakePollAnswer(poll_id=poll_id, option_ids=[0]))
    await bot.poll_answer_handler(update, context)

    text = context.bot.sent_messages[0]["text"]
    # 250 * 8 / 4 = 500
    assert "- 500 g Farine" in text


async def test_poll_answer_cleans_up_state() -> None:
    context = FakeContext()
    group = _setup_poll_group(context)
    poll_id = group["poll_ids"][0]

    update = FakeUpdate(poll_answer=FakePollAnswer(poll_id=poll_id, option_ids=[0]))
    await bot.poll_answer_handler(update, context)

    assert context.bot_data["polls"] == {}


# --- timeout --------------------------------------------------------------


async def test_poll_timeout_sends_partial_list(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_bot = FakeBot()
    ingredients = make_fake_ingredients()
    group: dict[str, Any] = {
        "chat_id": 42,
        "ingredients": ingredients,
        "servings": 6,
        "original_servings": 6,
        "poll_ids": ["p1"],
        "option_maps": {"p1": {0: 0, 1: 1}},
        "selected_indices": {0},  # only first ingredient selected
        "answered_polls": set(),
        "timeout_task": None,
        "done": False,
    }
    bot_data: dict[str, Any] = {"polls": {"p1": group}}

    async def _instant_sleep(_: float) -> None:
        pass

    import bot as bot_module

    monkeypatch.setattr(bot_module.asyncio, "sleep", _instant_sleep)
    await bot._poll_timeout(fake_bot, group, bot_data)

    assert group["done"] is True
    assert len(fake_bot.sent_messages) == 1
    text = fake_bot.sent_messages[0]["text"]
    assert text.startswith(bot.MSG_COURSES_TIMEOUT)
    assert "Pommes" in text


async def test_poll_timeout_no_selections(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_bot = FakeBot()
    group: dict[str, Any] = {
        "chat_id": 42,
        "ingredients": make_fake_ingredients(),
        "servings": 6,
        "original_servings": 6,
        "poll_ids": ["p1"],
        "option_maps": {},
        "selected_indices": set(),
        "answered_polls": set(),
        "timeout_task": None,
        "done": False,
    }
    bot_data: dict[str, Any] = {"polls": {"p1": group}}

    async def _instant_sleep(_: float) -> None:
        pass

    import bot as bot_module

    monkeypatch.setattr(bot_module.asyncio, "sleep", _instant_sleep)
    await bot._poll_timeout(fake_bot, group, bot_data)

    text = fake_bot.sent_messages[0]["text"]
    assert text == bot.MSG_COURSES_TIMEOUT


async def test_poll_timeout_noop_when_done(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_bot = FakeBot()
    group: dict[str, Any] = {
        "chat_id": 42,
        "ingredients": [],
        "servings": 4,
        "original_servings": 4,
        "poll_ids": [],
        "option_maps": {},
        "selected_indices": set(),
        "answered_polls": set(),
        "timeout_task": None,
        "done": True,  # already completed
    }

    async def _instant_sleep(_: float) -> None:
        pass

    import bot as bot_module

    monkeypatch.setattr(bot_module.asyncio, "sleep", _instant_sleep)
    await bot._poll_timeout(fake_bot, group, {})

    assert len(fake_bot.sent_messages) == 0


# --- helper functions -----------------------------------------------------


def test_format_ingredient_line_full() -> None:
    ing = Ingredient(id=1, recipe_id=1, name="Farine", quantity=500.0, unit="g")
    assert format_ingredient_line(ing.name, ing.quantity, ing.unit, 4, 4) == "500 g Farine"


def test_format_ingredient_line_no_quantity() -> None:
    ing = Ingredient(id=1, recipe_id=1, name="Sel", quantity=None, unit=None)
    assert format_ingredient_line(ing.name, ing.quantity, ing.unit, 4, 4) == "Sel"


def test_format_ingredient_line_no_unit() -> None:
    ing = Ingredient(id=1, recipe_id=1, name="Oeufs", quantity=3.0, unit=None)
    assert format_ingredient_line(ing.name, ing.quantity, ing.unit, 6, 6) == "3 Oeufs"


def test_format_ingredient_line_adjusted() -> None:
    ing = Ingredient(id=1, recipe_id=1, name="Farine", quantity=250.0, unit="g")
    # 250 * 8 / 4 = 500
    assert format_ingredient_line(ing.name, ing.quantity, ing.unit, 8, 4) == "500 g Farine"


def test_build_shopping_list_normal() -> None:
    ingredients = [
        Ingredient(id=1, recipe_id=1, name="Farine", quantity=500.0, unit="g"),
        Ingredient(id=2, recipe_id=1, name="Sel", quantity=None, unit=None),
    ]
    text = bot._build_shopping_list(ingredients, 4, 4)
    assert text == f"{bot.MSG_COURSES_HEADER}\n- 500 g Farine\n- Sel"


def test_build_shopping_list_empty() -> None:
    assert bot._build_shopping_list([], 4, 4) == bot.MSG_COURSES_EMPTY


def test_build_shopping_list_timeout_with_items() -> None:
    ingredients = [
        Ingredient(id=1, recipe_id=1, name="Farine", quantity=500.0, unit="g"),
    ]
    text = bot._build_shopping_list(ingredients, 4, 4, timed_out=True)
    assert text.startswith(bot.MSG_COURSES_TIMEOUT)
    assert "- 500 g Farine" in text


def test_build_shopping_list_timeout_empty() -> None:
    text = bot._build_shopping_list([], 4, 4, timed_out=True)
    assert text == bot.MSG_COURSES_TIMEOUT


# --- error handler --------------------------------------------------------


async def test_error_handler_replies_generic_message(patch_update: None) -> None:
    update = FakeUpdate()
    context = FakeContext(error=RuntimeError("boom"))

    await bot.error_handler(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_GENERIC_ERROR, {})]


async def test_error_handler_handles_none_update(patch_update: None) -> None:
    context = FakeContext(error=RuntimeError("job blew up"))

    # Must not raise: job-queue errors call the handler with update=None.
    await bot.error_handler(None, context)


async def test_error_handler_swallows_reply_failure(patch_update: None) -> None:
    fake_message = FakeMessage(raise_on_reply=ConnectionError("network down"))
    update = FakeUpdate(effective_message=fake_message)
    context = FakeContext(error=RuntimeError("upstream"))

    # Must not raise: a failing reply must not recurse into the error handler.
    await bot.error_handler(update, context)

    assert fake_message.replies == []


async def test_error_handler_ignores_non_update_object(patch_update: None) -> None:
    # Background-job errors pass an arbitrary object as the first arg.
    context = FakeContext(error=RuntimeError("job"))

    # Must not raise.
    await bot.error_handler({"not": "an update"}, context)


# --- categorised error handler ------------------------------------------------


async def test_error_handler_categorizes_agent_error(patch_update: None) -> None:
    update = FakeUpdate()
    context = FakeContext(error=RecipeExtractionAgentError("sdk failure"))

    await bot.error_handler(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_AJOUTER_AGENT_ERROR, {})]


async def test_error_handler_categorizes_agent_timeout(patch_update: None) -> None:
    update = FakeUpdate()
    context = FakeContext(error=RecipeExtractionTimeout("timed out"))

    await bot.error_handler(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_AJOUTER_AGENT_ERROR, {})]


async def test_error_handler_categorizes_agent_content(patch_update: None) -> None:
    update = FakeUpdate()
    context = FakeContext(error=RecipeExtractionContentError("site bloqué"))

    await bot.error_handler(update, context)

    assert update.effective_message is not None
    # Global handler uses MSG_AJOUTER_VALIDATION_ERROR (no formatting) as fallback.
    assert update.effective_message.replies == [(bot.MSG_AJOUTER_VALIDATION_ERROR, {})]


async def test_error_handler_categorizes_agent_validation(patch_update: None) -> None:
    update = FakeUpdate()
    context = FakeContext(error=RecipeExtractionValidationError("bad schema"))

    await bot.error_handler(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_AJOUTER_VALIDATION_ERROR, {})]


async def test_error_handler_categorizes_embedding_error(patch_update: None) -> None:
    update = FakeUpdate()
    context = FakeContext(error=EmbeddingError("encode failed"))

    await bot.error_handler(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_CHERCHER_EMBEDDING_ERROR, {})]


async def test_error_handler_categorizes_pdf_error(patch_update: None) -> None:
    update = FakeUpdate()
    context = FakeContext(error=PdfGenerationError("render failed"))

    await bot.error_handler(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_RECIPE_PDF_ERROR, {})]


async def test_error_handler_categorizes_network_error(patch_update: None) -> None:
    update = FakeUpdate()
    context = FakeContext(error=TelegramNetworkError("connection reset"))

    await bot.error_handler(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_NETWORK_ERROR, {})]


async def test_error_handler_categorizes_database_error(patch_update: None) -> None:
    update = FakeUpdate()
    context = FakeContext(error=SAOperationalError("db", {}, Exception("conn refused")))

    await bot.error_handler(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_AJOUTER_DB_ERROR, {})]


async def test_error_handler_unknown_still_generic(patch_update: None) -> None:
    """Unknown exception types still get the generic fallback message."""
    update = FakeUpdate()
    context = FakeContext(error=ValueError("something unexpected"))

    await bot.error_handler(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_GENERIC_ERROR, {})]


# --- build_application() smoke -------------------------------------------


def test_build_application_registers_all_commands() -> None:
    app = bot.build_application()

    # Commands inside the ConversationHandler entry_points.
    conv_commands: set[str] = set()
    # Standalone CommandHandlers.
    standalone_commands: set[str] = set()

    for handler_group in app.handlers.values():
        for handler in handler_group:
            if isinstance(handler, ConversationHandler):
                for ep in handler.entry_points:
                    if isinstance(ep, CommandHandler):
                        conv_commands.update(ep.commands)
            elif isinstance(handler, CommandHandler):
                standalone_commands.update(handler.commands)

    assert {"ajouter", "chercher", "personnes"} <= conv_commands
    assert "courses" in standalone_commands


def test_build_application_registers_callback_handler() -> None:
    app = bot.build_application()

    has_callback = any(
        isinstance(handler, CallbackQueryHandler)
        for group in app.handlers.values()
        for handler in group
    )
    assert has_callback


def test_build_application_registers_poll_answer_handler() -> None:
    app = bot.build_application()

    has_poll = any(
        isinstance(handler, PollAnswerHandler)
        for group in app.handlers.values()
        for handler in group
    )
    assert has_poll


def test_build_application_registers_error_handler() -> None:
    app = bot.build_application()

    assert bot.error_handler in app.error_handlers


def test_build_application_uses_settings_token() -> None:
    app = bot.build_application()

    assert app.bot.token == get_settings().telegram_bot_token


# --- ConversationHandler: return values ------------------------------------


async def test_ajouter_with_args_returns_end(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ajouter_deps(monkeypatch)
    update = FakeUpdate()
    context = FakeContext(args=["https://example.com/recette"])

    result = await bot.ajouter_command(update, context)

    assert result == ConversationHandler.END


async def test_chercher_with_args_returns_end(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_chercher_deps(monkeypatch, search_results=[])
    update = FakeUpdate()
    context = FakeContext(args=["tarte"])

    result = await bot.chercher_command(update, context)

    assert result == ConversationHandler.END


async def test_personnes_with_args_returns_end(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = make_fake_recipe(id=1, servings=6)
    _patch_callback_deps(monkeypatch, recipe=recipe)
    update = FakeUpdate()
    context = FakeContext(
        args=["4"],
        user_data={bot.USER_DATA_SELECTED_RECIPE_ID: 1},
    )

    result = await bot.personnes_command(update, context)

    assert result == ConversationHandler.END


# --- ConversationHandler: receive handlers ---------------------------------


async def test_ajouter_receive_url_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ajouter_deps(monkeypatch)
    msg = FakeMessage(text="https://example.com/tarte")
    update = FakeUpdate(effective_message=msg)
    context = FakeContext()

    result = await bot._ajouter_receive_url(update, context)

    assert result == ConversationHandler.END
    texts = [t for t, _ in msg.replies]
    assert texts[0] == bot.MSG_AJOUTER_ACK
    assert "Tarte aux pommes" in texts[1]


async def test_ajouter_receive_url_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ajouter_deps(monkeypatch)
    msg = FakeMessage(text="not-a-url")
    update = FakeUpdate(effective_message=msg)
    context = FakeContext()

    result = await bot._ajouter_receive_url(update, context)

    assert result == ConversationHandler.END
    assert msg.replies == [(bot.MSG_AJOUTER_INVALID_URL, {})]


async def test_ajouter_receive_url_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ajouter_deps(monkeypatch)
    msg = FakeMessage(text="   ")
    update = FakeUpdate(effective_message=msg)
    context = FakeContext()

    result = await bot._ajouter_receive_url(update, context)

    assert result == ConversationHandler.END
    assert msg.replies == [(bot.MSG_AJOUTER_USAGE, {})]


async def test_chercher_receive_query(monkeypatch: pytest.MonkeyPatch) -> None:
    recipes = [make_fake_recipe(id=1, title="Tarte aux pommes", servings=6)]
    _patch_chercher_deps(monkeypatch, search_results=recipes)
    msg = FakeMessage(text="tarte")
    update = FakeUpdate(effective_message=msg)
    context = FakeContext()

    result = await bot._chercher_receive_query(update, context)

    assert result == ConversationHandler.END
    texts = [t for t, _ in msg.replies]
    assert texts[0] == bot.MSG_CHERCHER_ACK


async def test_chercher_receive_query_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_chercher_deps(monkeypatch, search_results=[])
    msg = FakeMessage(text="   ")
    update = FakeUpdate(effective_message=msg)
    context = FakeContext()

    result = await bot._chercher_receive_query(update, context)

    assert result == ConversationHandler.END
    assert msg.replies == [(bot.MSG_CHERCHER_USAGE, {})]


async def test_personnes_receive_servings_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = make_fake_recipe(id=1, servings=6)
    _patch_callback_deps(monkeypatch, recipe=recipe)
    msg = FakeMessage(text="4")
    update = FakeUpdate(effective_message=msg)
    context = FakeContext(
        user_data={bot.USER_DATA_SELECTED_RECIPE_ID: 1},
    )

    result = await bot._personnes_receive_servings(update, context)

    assert result == ConversationHandler.END
    texts = [t for t, _ in msg.replies]
    assert bot.MSG_PERSONNES_ACK.format(n=4, original_servings=6) in texts


async def test_personnes_receive_servings_invalid() -> None:
    msg = FakeMessage(text="abc")
    update = FakeUpdate(effective_message=msg)
    context = FakeContext(
        user_data={bot.USER_DATA_SELECTED_RECIPE_ID: 1},
    )

    result = await bot._personnes_receive_servings(update, context)

    assert result == ConversationHandler.END
    assert msg.replies == [(bot.MSG_PERSONNES_INVALID, {})]


async def test_personnes_receive_servings_no_recipe() -> None:
    msg = FakeMessage(text="4")
    update = FakeUpdate(effective_message=msg)
    context = FakeContext(user_data={})

    result = await bot._personnes_receive_servings(update, context)

    assert result == ConversationHandler.END
    assert msg.replies == [(bot.MSG_PERSONNES_NO_RECIPE, {})]


# --- /annuler --------------------------------------------------------------


async def test_annuler_command() -> None:
    update = FakeUpdate()
    context = FakeContext()

    result = await bot.annuler_command(update, context)

    assert result == ConversationHandler.END
    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_ANNULER, {})]


# --- build_application: ConversationHandler --------------------------------


def test_build_application_registers_conversation_handler() -> None:
    app = bot.build_application()

    conv_handlers = [
        handler
        for group in app.handlers.values()
        for handler in group
        if isinstance(handler, ConversationHandler)
    ]
    assert len(conv_handlers) == 1

    conv = conv_handlers[0]
    entry_commands = set()
    for ep in conv.entry_points:
        if isinstance(ep, CommandHandler):
            entry_commands.update(ep.commands)
    assert {"ajouter", "chercher", "personnes"} == entry_commands

    assert bot.STATE_AWAITING_URL in conv.states
    assert bot.STATE_AWAITING_QUERY in conv.states
    assert bot.STATE_AWAITING_SERVINGS in conv.states

    fallback_commands = set()
    for fb in conv.fallbacks:
        if isinstance(fb, CommandHandler):
            fallback_commands.update(fb.commands)
    assert "annuler" in fallback_commands


# --- access-control gate ---------------------------------------------------


def _patch_access_control(
    monkeypatch: pytest.MonkeyPatch,
    *,
    user_ids: frozenset[int] = frozenset(),
    group_ids: frozenset[int] = frozenset(),
) -> None:
    """Override access-control settings for a single test."""
    monkeypatch.setenv("ALLOWED_USER_IDS", ",".join(str(i) for i in user_ids))
    monkeypatch.setenv("ALLOWED_GROUP_IDS", ",".join(str(i) for i in group_ids))
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_access_gate_allows_when_no_restriction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No allow-lists configured → gate does not block."""
    update = FakeUpdate()
    ctx = FakeContext()
    # All allow-lists empty by default — should pass through without raising.
    await bot._access_control_gate(update, ctx)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_access_gate_allows_authorized_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_access_control(monkeypatch, user_ids=frozenset({42}))
    update = FakeUpdate(effective_user=FakeUser(id=42))
    ctx = FakeContext()
    await bot._access_control_gate(update, ctx)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_access_gate_blocks_unauthorized_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_access_control(monkeypatch, user_ids=frozenset({99}))
    update = FakeUpdate(effective_user=FakeUser(id=42))
    ctx = FakeContext()
    with pytest.raises(ApplicationHandlerStop):
        await bot._access_control_gate(update, ctx)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_access_gate_allows_authorized_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_access_control(monkeypatch, group_ids=frozenset({-100123}))
    update = FakeUpdate(
        effective_user=FakeUser(id=42),
        effective_chat=FakeChat(id=-100123),
    )
    ctx = FakeContext()
    await bot._access_control_gate(update, ctx)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_access_gate_user_in_allowed_group_not_in_allowed_users(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OR logic: group membership is sufficient even when user_ids is empty."""
    _patch_access_control(monkeypatch, group_ids=frozenset({-100123}))
    update = FakeUpdate(
        effective_user=FakeUser(id=99),
        effective_chat=FakeChat(id=-100123),
    )
    ctx = FakeContext()
    await bot._access_control_gate(update, ctx)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_access_gate_blocks_when_no_user_no_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When access control is active but IDs cannot be extracted → deny."""
    _patch_access_control(monkeypatch, user_ids=frozenset({42}))
    update = FakeUpdate(effective_user=None, effective_chat=None)
    ctx = FakeContext()
    with pytest.raises(ApplicationHandlerStop):
        await bot._access_control_gate(update, ctx)  # type: ignore[arg-type]


def test_build_application_no_access_control_no_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no allow-lists → no TypeHandler in group -1."""
    app = bot.build_application()
    group_minus_1 = app.handlers.get(-1, [])
    assert not any(isinstance(h, TypeHandler) for h in group_minus_1)


def test_build_application_with_access_control_registers_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_access_control(monkeypatch, user_ids=frozenset({123}))
    app = bot.build_application()
    group_minus_1 = app.handlers.get(-1, [])
    assert any(isinstance(h, TypeHandler) for h in group_minus_1)


# --- command menu registration --------------------------------------------


def test_build_application_wires_post_init_for_command_menu() -> None:
    """The Application must invoke ``_post_init`` on startup so the command
    menu is pushed to Telegram before polling begins."""
    app = bot.build_application()
    assert app.post_init is bot._post_init


async def test_post_init_sets_commands_for_private_and_group_scopes() -> None:
    """``_post_init`` must register the full command list under both the
    private-chat scope and the group-chat scope so the menu appears everywhere
    the bot is used."""
    application = AsyncMock()
    application.bot = AsyncMock()

    await bot._post_init(application)

    assert application.bot.set_my_commands.await_count == 2
    calls = application.bot.set_my_commands.await_args_list

    scopes = {type(call.kwargs["scope"]) for call in calls}
    assert scopes == {BotCommandScopeAllPrivateChats, BotCommandScopeAllGroupChats}

    for call in calls:
        commands = call.args[0]
        assert all(isinstance(c, BotCommand) for c in commands)
        assert [c.command for c in commands] == [
            "ajouter",
            "chercher",
            "personnes",
            "courses",
            "annuler",
        ]
        assert all(c.description.strip() for c in commands)
