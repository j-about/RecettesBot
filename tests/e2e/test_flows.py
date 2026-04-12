"""End-to-end tests exercising complete user journeys through the bot.

Each test simulates a real user interacting with the Telegram bot:
constructing fake Update/Context objects, calling the handler, and
asserting on the reply sequence. All external dependencies (agent SDK,
embedding model, database, PDF generator) are replaced by fakes wired
up in the autouse ``patch_bot_deps`` fixture (see ``conftest.py``).

The crown jewel is ``test_complete_chain`` which exercises the full
``/ajouter → /chercher → select → /personnes → /courses → poll answer``
flow in a single test with shared state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

import bot
from agent import (
    RecipeExtractionAgentError,
    RecipeExtractionTimeout,
    RecipeExtractionValidationError,
)
from embedding import EmbeddingEncodeError
from tests.e2e.conftest import InMemoryDatabase
from tests.mocks.stubs import stub_encode_text, stub_extract_recipe
from tests.mocks.telegram import (
    FakeCallbackQuery,
    FakeContext,
    FakeMessage,
    FakePollAnswer,
    FakeUpdate,
)

# === /ajouter flow ===========================================================


async def test_ajouter_persists_recipe_and_replies_success(
    fake_db: InMemoryDatabase,
) -> None:
    """Full /ajouter flow: ACK → extract → embed → persist → success message."""
    update = FakeUpdate()
    context = FakeContext(args=["https://example.com/tarte"])

    await bot.ajouter_command(update, context)

    msg = update.effective_message
    assert msg is not None
    texts = [t for t, _ in msg.replies]
    assert texts[0] == bot.MSG_AJOUTER_ACK
    assert "Tarte aux pommes" in texts[1]
    assert "6 personnes" in texts[1]

    # Recipe was persisted in the fake DB
    assert len(fake_db.recipes) == 1
    recipe = next(iter(fake_db.recipes.values()))
    assert recipe.title == "Tarte aux pommes"
    assert recipe.servings == 6
    assert len(recipe.ingredients) == 2


# === /chercher flow ===========================================================


async def test_chercher_returns_inline_keyboard(
    fake_db: InMemoryDatabase,
) -> None:
    """Ajouter a recipe, then /chercher should find it and show an inline keyboard."""
    # Setup: add a recipe first
    add_update = FakeUpdate()
    add_ctx = FakeContext(args=["https://example.com/tarte"])
    await bot.ajouter_command(add_update, add_ctx)
    assert len(fake_db.recipes) == 1

    # Now search
    search_update = FakeUpdate()
    search_ctx = FakeContext(args=["tarte"])
    await bot.chercher_command(search_update, search_ctx)

    msg = search_update.effective_message
    assert msg is not None
    texts = [t for t, _ in msg.replies]
    assert texts[0] == bot.MSG_CHERCHER_ACK
    assert texts[1] == "📋 Résultats :"
    _, kwargs = msg.replies[1]
    markup = kwargs["reply_markup"]
    assert len(markup.inline_keyboard) == 1
    btn = markup.inline_keyboard[0][0]
    assert "Tarte aux pommes" in btn.text


# === callback (recipe selected) flow =========================================


async def test_recipe_selected_sends_pdf(
    fake_db: InMemoryDatabase,
) -> None:
    """Select a recipe from /chercher results → receive a PDF document."""
    # Add a recipe
    add_update = FakeUpdate()
    await bot.ajouter_command(add_update, FakeContext(args=["https://example.com/tarte"]))
    recipe_id = next(iter(fake_db.recipes))

    # Select it via callback
    cb_msg = FakeMessage()
    query = FakeCallbackQuery(data=str(recipe_id), message=cb_msg)
    update = FakeUpdate(callback_query=query)
    context = FakeContext()

    await bot.recipe_selected_callback(update, context)

    assert len(cb_msg.documents) == 1
    assert cb_msg.documents[0]["document"] == b"%PDF-e2e-test"
    assert cb_msg.documents[0]["filename"] == "Tarte aux pommes.pdf"
    # user_data should have the selected recipe ID
    assert context.user_data is not None
    assert context.user_data[bot.USER_DATA_SELECTED_RECIPE_ID] == recipe_id


# === /personnes flow ==========================================================


async def test_personnes_adjusts_servings_and_sends_pdf(
    fake_db: InMemoryDatabase,
) -> None:
    """Select a recipe, then /personnes 8 → confirm adjustment + send PDF."""
    # Add recipe
    add_update = FakeUpdate()
    await bot.ajouter_command(add_update, FakeContext(args=["https://example.com/tarte"]))
    recipe_id = next(iter(fake_db.recipes))

    # Set up user_data as if recipe was selected
    update = FakeUpdate()
    context = FakeContext(
        args=["8"],
        user_data={bot.USER_DATA_SELECTED_RECIPE_ID: recipe_id},
    )

    await bot.personnes_command(update, context)

    msg = update.effective_message
    assert msg is not None
    texts = [t for t, _ in msg.replies]
    assert "8" in texts[0]
    assert context.user_data is not None
    assert context.user_data[bot.USER_DATA_ADJUSTED_SERVINGS] == 8
    # PDF sent
    assert len(msg.documents) == 1


# === /courses → poll answer flow ==============================================


async def test_courses_sends_poll_and_collects_answer(
    fake_db: InMemoryDatabase,
) -> None:
    """Full /courses flow: send poll → answer poll → receive shopping list."""
    # Add recipe
    add_update = FakeUpdate()
    await bot.ajouter_command(add_update, FakeContext(args=["https://example.com/tarte"]))
    recipe_id = next(iter(fake_db.recipes))

    # Launch /courses
    courses_update = FakeUpdate()
    courses_ctx = FakeContext(
        user_data={bot.USER_DATA_SELECTED_RECIPE_ID: recipe_id},
    )
    await bot.courses_command(courses_update, courses_ctx)

    msg = courses_update.effective_message
    assert msg is not None
    texts = [t for t, _ in msg.replies]
    assert texts[0] == bot.MSG_COURSES_ACK
    assert len(msg.polls) == 1
    assert msg.polls[0]["allows_multiple_answers"] is True

    # Answer the poll — select first ingredient
    poll_id = list(courses_ctx.bot_data["polls"])[0]
    answer_update = FakeUpdate(
        poll_answer=FakePollAnswer(poll_id=poll_id, option_ids=[0]),
    )
    await bot.poll_answer_handler(answer_update, courses_ctx)

    # Shopping list sent
    assert len(courses_ctx.bot.sent_messages) == 1
    text = courses_ctx.bot.sent_messages[0]["text"]
    assert text.startswith(bot.MSG_COURSES_HEADER)
    assert "Pommes" in text


# === Complete chain ===========================================================


async def test_complete_chain(
    fake_db: InMemoryDatabase,
) -> None:
    """/ajouter → /chercher → select → /personnes → /courses → poll answer.

    This is the full user journey: add a recipe from a URL, search for it,
    select it, adjust servings, generate a shopping list poll, and answer it.
    All state flows through ``context.user_data`` and the in-memory DB.
    """
    shared_user_data: dict[str, Any] = {}

    # 1. /ajouter https://example.com/tarte
    add_update = FakeUpdate()
    add_ctx = FakeContext(args=["https://example.com/tarte"])
    await bot.ajouter_command(add_update, add_ctx)

    assert len(fake_db.recipes) == 1
    add_texts = [t for t, _ in add_update.effective_message.replies]
    assert add_texts[0] == bot.MSG_AJOUTER_ACK
    assert "Tarte aux pommes" in add_texts[1]

    # 2. /chercher tarte
    search_update = FakeUpdate()
    search_ctx = FakeContext(args=["tarte"])
    await bot.chercher_command(search_update, search_ctx)

    search_texts = [t for t, _ in search_update.effective_message.replies]
    assert search_texts[0] == bot.MSG_CHERCHER_ACK
    _, search_kwargs = search_update.effective_message.replies[1]
    btn = search_kwargs["reply_markup"].inline_keyboard[0][0]
    recipe_id_str = btn.callback_data

    # 3. Select the recipe (callback)
    select_msg = FakeMessage()
    select_query = FakeCallbackQuery(data=recipe_id_str, message=select_msg)
    select_update = FakeUpdate(callback_query=select_query)
    select_ctx = FakeContext(user_data=shared_user_data)
    await bot.recipe_selected_callback(select_update, select_ctx)

    assert len(select_msg.documents) == 1
    assert shared_user_data[bot.USER_DATA_SELECTED_RECIPE_ID] == int(recipe_id_str)

    # 4. /personnes 10
    personnes_update = FakeUpdate()
    personnes_ctx = FakeContext(args=["10"], user_data=shared_user_data)
    await bot.personnes_command(personnes_update, personnes_ctx)

    personnes_texts = [t for t, _ in personnes_update.effective_message.replies]
    assert "10" in personnes_texts[0]
    assert shared_user_data[bot.USER_DATA_ADJUSTED_SERVINGS] == 10
    assert len(personnes_update.effective_message.documents) == 1

    # 5. /courses
    courses_update = FakeUpdate()
    courses_ctx = FakeContext(user_data=shared_user_data)
    await bot.courses_command(courses_update, courses_ctx)

    courses_texts = [t for t, _ in courses_update.effective_message.replies]
    assert courses_texts[0] == bot.MSG_COURSES_ACK
    assert len(courses_update.effective_message.polls) >= 1

    # 6. Answer poll — select both ingredients
    poll_id = list(courses_ctx.bot_data["polls"])[0]
    poll_update = FakeUpdate(
        poll_answer=FakePollAnswer(poll_id=poll_id, option_ids=[0, 1]),
    )
    await bot.poll_answer_handler(poll_update, courses_ctx)

    assert len(courses_ctx.bot.sent_messages) == 1
    shopping_list = courses_ctx.bot.sent_messages[0]["text"]
    assert shopping_list.startswith(bot.MSG_COURSES_HEADER)
    assert "Pommes" in shopping_list
    assert "Sucre" in shopping_list


# === Error scenarios ==========================================================


async def test_ajouter_agent_timeout_replies_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent timeout → user gets the French error message."""
    monkeypatch.setattr(
        bot,
        "extract_recipe",
        stub_extract_recipe(raise_exc=RecipeExtractionTimeout("timed out")),
    )

    update = FakeUpdate()
    context = FakeContext(args=["https://example.com/slow-recipe"])
    await bot.ajouter_command(update, context)

    texts = [t for t, _ in update.effective_message.replies]
    assert texts == [bot.MSG_AJOUTER_ACK, bot.MSG_AJOUTER_AGENT_ERROR]


async def test_ajouter_validation_error_replies_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent returns invalid data → user gets the French error message."""
    monkeypatch.setattr(
        bot,
        "extract_recipe",
        stub_extract_recipe(raise_exc=RecipeExtractionValidationError("bad")),
    )

    update = FakeUpdate()
    context = FakeContext(args=["https://example.com/not-a-recipe"])
    await bot.ajouter_command(update, context)

    texts = [t for t, _ in update.effective_message.replies]
    assert texts == [bot.MSG_AJOUTER_ACK, bot.MSG_AJOUTER_VALIDATION_ERROR]


async def test_ajouter_agent_error_replies_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent SDK error → user gets the French error message."""
    monkeypatch.setattr(
        bot,
        "extract_recipe",
        stub_extract_recipe(raise_exc=RecipeExtractionAgentError("SDK failed")),
    )

    update = FakeUpdate()
    context = FakeContext(args=["https://example.com/broken"])
    await bot.ajouter_command(update, context)

    texts = [t for t, _ in update.effective_message.replies]
    assert texts == [bot.MSG_AJOUTER_ACK, bot.MSG_AJOUTER_AGENT_ERROR]


async def test_chercher_embedding_error_replies_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding encode failure → user gets the French error message."""
    monkeypatch.setattr(
        bot,
        "encode_text",
        stub_encode_text(raise_exc=EmbeddingEncodeError("model exploded")),
    )

    update = FakeUpdate()
    context = FakeContext(args=["tarte"])
    await bot.chercher_command(update, context)

    texts = [t for t, _ in update.effective_message.replies]
    assert texts == [bot.MSG_CHERCHER_ACK, bot.MSG_CHERCHER_EMBEDDING_ERROR]


async def test_ajouter_db_error_replies_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Database write failure → user gets the French error message."""

    @asynccontextmanager
    async def _fail() -> AsyncIterator[Any]:
        raise RuntimeError("db down")
        yield  # pragma: no cover

    monkeypatch.setattr(bot, "session_scope", _fail)

    update = FakeUpdate()
    context = FakeContext(args=["https://example.com/tarte"])
    await bot.ajouter_command(update, context)

    texts = [t for t, _ in update.effective_message.replies]
    assert texts == [bot.MSG_AJOUTER_ACK, bot.MSG_AJOUTER_DB_ERROR]


async def test_error_handler_categorises_and_replies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global error handler → categorises exception → French user message."""
    from telegram.error import NetworkError as TelegramNetworkError

    update = FakeUpdate()
    context = FakeContext(error=TelegramNetworkError("connection reset"))

    await bot.error_handler(update, context)

    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_NETWORK_ERROR, {})]


# === ConversationHandler flows ================================================


async def test_ajouter_conversation_flow(
    fake_db: InMemoryDatabase,
) -> None:
    """/ajouter (no arg) → prompt → user sends URL → recipe persisted."""
    # Step 1: /ajouter with no argument
    update = FakeUpdate()
    context = FakeContext(args=[])

    result = await bot.ajouter_command(update, context)

    assert result == bot.STATE_AWAITING_URL
    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_AJOUTER_PROMPT, {})]

    # Step 2: user sends the URL as a plain text message
    url_msg = FakeMessage(text="https://example.com/tarte")
    url_update = FakeUpdate(effective_message=url_msg)
    url_context = FakeContext()

    result = await bot._ajouter_receive_url(url_update, url_context)

    assert result == bot.ConversationHandler.END
    texts = [t for t, _ in url_msg.replies]
    assert texts[0] == bot.MSG_AJOUTER_ACK
    assert "Tarte aux pommes" in texts[1]
    assert len(fake_db.recipes) == 1


async def test_chercher_conversation_flow(
    fake_db: InMemoryDatabase,
) -> None:
    """Add recipe, /chercher (no arg) → prompt → query → results."""
    # Setup: add a recipe first
    add_update = FakeUpdate()
    await bot.ajouter_command(add_update, FakeContext(args=["https://example.com/tarte"]))
    assert len(fake_db.recipes) == 1

    # Step 1: /chercher with no argument
    update = FakeUpdate()
    context = FakeContext(args=[])

    result = await bot.chercher_command(update, context)

    assert result == bot.STATE_AWAITING_QUERY
    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_CHERCHER_PROMPT, {})]

    # Step 2: user sends the query as a plain text message
    query_msg = FakeMessage(text="tarte")
    query_update = FakeUpdate(effective_message=query_msg)
    query_context = FakeContext()

    result = await bot._chercher_receive_query(query_update, query_context)

    assert result == bot.ConversationHandler.END
    texts = [t for t, _ in query_msg.replies]
    assert texts[0] == bot.MSG_CHERCHER_ACK
    assert texts[1] == "📋 Résultats :"


async def test_personnes_conversation_flow(
    fake_db: InMemoryDatabase,
) -> None:
    """/personnes (no arg) → prompt → user sends number → adjustment + PDF."""
    # Setup: add a recipe and set it as selected
    add_update = FakeUpdate()
    await bot.ajouter_command(add_update, FakeContext(args=["https://example.com/tarte"]))
    recipe_id = next(iter(fake_db.recipes))

    # Step 1: /personnes with no argument
    update = FakeUpdate()
    context = FakeContext(
        args=[],
        user_data={bot.USER_DATA_SELECTED_RECIPE_ID: recipe_id},
    )

    result = await bot.personnes_command(update, context)

    assert result == bot.STATE_AWAITING_SERVINGS
    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_PERSONNES_PROMPT, {})]

    # Step 2: user sends the number as a plain text message
    num_msg = FakeMessage(text="8")
    num_update = FakeUpdate(effective_message=num_msg)
    num_context = FakeContext(
        user_data={bot.USER_DATA_SELECTED_RECIPE_ID: recipe_id},
    )

    result = await bot._personnes_receive_servings(num_update, num_context)

    assert result == bot.ConversationHandler.END
    texts = [t for t, _ in num_msg.replies]
    assert "8" in texts[0]
    assert len(num_msg.documents) == 1


async def test_annuler_ends_conversation() -> None:
    """/annuler → cancel message, conversation ends."""
    update = FakeUpdate()
    context = FakeContext()

    result = await bot.annuler_command(update, context)

    assert result == bot.ConversationHandler.END
    assert update.effective_message is not None
    assert update.effective_message.replies == [(bot.MSG_ANNULER, {})]
