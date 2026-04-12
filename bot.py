"""Telegram bot for RecettesBot.

This module wires up the python-telegram-bot ``Application`` with the command
handlers:

- ``/ajouter`` — extract a recipe from a URL via the Claude Agent SDK, compute
  a title embedding, and persist both to the database.
- ``/chercher`` — semantic search over stored recipes using vector similarity.
- ``/personnes`` — adjust the serving count and regenerate the recipe PDF.
- ``/courses`` — generate a shopping-list poll from the selected recipe's
  ingredients.
- ``/annuler`` — abort the current conversation prompt.

Additional infrastructure:

- Rich error handler categorising network, agent, embedding, PDF, database,
  and unknown errors — never exposes stack traces to the user.
- Optional access-control gate (handler group ``-1``) restricting usage to
  configured user / group / channel allow-lists.
- Logfire observability spans for every handler.
- Polling-only entry point; webhook mode is not supported.
"""

import asyncio
import contextlib
import json
import logging
from typing import Any

import logfire
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import select
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import NetworkError as TelegramNetworkError
from telegram.ext import (
    Application,
    ApplicationHandlerStop,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    PollAnswerHandler,
    TypeHandler,
    filters,
)

from agent import (
    RecipeExtractionAgentError,
    RecipeExtractionContentError,
    RecipeExtractionTimeout,
    RecipeExtractionValidationError,
    extract_recipe,
)
from db import session_scope
from embedding import EmbeddingError, encode_text, get_embedding_model
from models import Ingredient, Recipe, format_ingredient_line
from pdf import PdfGenerationError, generate_recipe_pdf
from settings import get_settings

logger = logging.getLogger(__name__)

__all__ = [
    "MSG_AJOUTER_ACK",
    "MSG_AJOUTER_AGENT_ERROR",
    "MSG_AJOUTER_CONTENT_ERROR",
    "MSG_AJOUTER_DB_ERROR",
    "MSG_AJOUTER_INVALID_URL",
    "MSG_AJOUTER_PROMPT",
    "MSG_AJOUTER_SUCCESS",
    "MSG_AJOUTER_USAGE",
    "MSG_AJOUTER_VALIDATION_ERROR",
    "MSG_ANNULER",
    "MSG_CHERCHER_ACK",
    "MSG_CHERCHER_DB_ERROR",
    "MSG_CHERCHER_EMBEDDING_ERROR",
    "MSG_CHERCHER_NO_RESULTS",
    "MSG_CHERCHER_PROMPT",
    "MSG_CHERCHER_USAGE",
    "MSG_RECIPE_NOT_FOUND",
    "MSG_RECIPE_PDF_ERROR",
    "MSG_RECIPE_SELECTED",
    "MSG_RECIPE_SEND_ERROR",
    "MSG_COURSES_ACK",
    "MSG_COURSES_DB_ERROR",
    "MSG_COURSES_EMPTY",
    "MSG_COURSES_HEADER",
    "MSG_COURSES_NO_INGREDIENTS",
    "MSG_COURSES_NO_RECIPE",
    "MSG_COURSES_POLL_QUESTION",
    "MSG_COURSES_POLL_QUESTION_MULTI",
    "MSG_COURSES_TIMEOUT",
    "MSG_GENERIC_ERROR",
    "MSG_NETWORK_ERROR",
    "MSG_PERSONNES_ACK",
    "MSG_PERSONNES_INVALID",
    "MSG_PERSONNES_NO_RECIPE",
    "MSG_PERSONNES_PROMPT",
    "MSG_PERSONNES_USAGE",
    "POLL_TIMEOUT_SECONDS",
    "STATE_AWAITING_QUERY",
    "STATE_AWAITING_SERVINGS",
    "STATE_AWAITING_URL",
    "USER_DATA_ADJUSTED_SERVINGS",
    "USER_DATA_SELECTED_RECIPE_ID",
    "ajouter_command",
    "annuler_command",
    "build_application",
    "chercher_command",
    "courses_command",
    "error_handler",
    "personnes_command",
    "poll_answer_handler",
    "recipe_selected_callback",
    "run",
]


# --- session-state keys ---------------------------------------------------
# Shared across /chercher callback, /personnes, and /courses so all handlers
# reference the same string keys without re-typing literals.
USER_DATA_SELECTED_RECIPE_ID = "selected_recipe_id"
USER_DATA_ADJUSTED_SERVINGS = "adjusted_servings"


# --- French user-facing messages -------------------------------------------
# Constants so the test suite can pin them as exact-match assertions.

# /ajouter
MSG_AJOUTER_USAGE = "❌ Utilisation : /ajouter <url>"
MSG_AJOUTER_INVALID_URL = "❌ L'URL doit commencer par http:// ou https://"
MSG_AJOUTER_ACK = "⏳ Je récupère la recette, un instant…"
MSG_AJOUTER_SUCCESS = "✅ Recette ajoutée : {title} ({servings} personnes)"
MSG_AJOUTER_AGENT_ERROR = "❌ Impossible de récupérer la recette. Vérifiez l'URL et réessayez."
MSG_AJOUTER_CONTENT_ERROR = "❌ Impossible d'extraire la recette : {reason}"
MSG_AJOUTER_DB_ERROR = "❌ Erreur interne lors de l'enregistrement. Réessayez plus tard."
MSG_AJOUTER_VALIDATION_ERROR = (
    "❌ Aucune recette n'a pu être extraite depuis cette page. "
    "Vérifiez que l'URL mène bien à une recette."
)

# /chercher
MSG_CHERCHER_USAGE = "❌ Utilisation : /chercher <texte>"
MSG_CHERCHER_ACK = "🔍 Recherche en cours…"
MSG_CHERCHER_NO_RESULTS = "🔍 Aucune recette trouvée pour « {query} »."
MSG_CHERCHER_DB_ERROR = "❌ Erreur interne lors de la recherche. Réessayez plus tard."
MSG_CHERCHER_EMBEDDING_ERROR = "❌ Erreur lors de l'encodage de la recherche. Réessayez plus tard."

# /chercher callback
MSG_RECIPE_SELECTED = "✅ Recette sélectionnée."
MSG_RECIPE_NOT_FOUND = "❌ Recette introuvable."
MSG_RECIPE_PDF_ERROR = "❌ Erreur lors de la génération du PDF."
MSG_RECIPE_SEND_ERROR = "❌ Erreur réseau. Réessayez plus tard."

# /personnes
MSG_PERSONNES_USAGE = "❌ Utilisation : /personnes <nombre>"
MSG_PERSONNES_INVALID = "❌ Le nombre de personnes doit être un entier positif."
MSG_PERSONNES_NO_RECIPE = "❌ Aucune recette sélectionnée. Utilisez /chercher d'abord."
MSG_PERSONNES_ACK = "👍 Nombre de personnes mis à jour : {n} (au lieu de {original_servings})."

# /courses
MSG_COURSES_NO_RECIPE = "❌ Aucune recette sélectionnée. Utilisez /chercher d'abord."
MSG_COURSES_ACK = "🛒 Préparation de la liste de courses…"
MSG_COURSES_NO_INGREDIENTS = "ℹ️ Cette recette n'a aucun ingrédient."
MSG_COURSES_DB_ERROR = "❌ Erreur interne. Réessayez plus tard."
MSG_COURSES_POLL_QUESTION = "Sélectionnez les ingrédients à acheter :"
MSG_COURSES_POLL_QUESTION_MULTI = "Sélectionnez les ingrédients ({start}-{end}) :"
MSG_COURSES_HEADER = "🛒 Liste de courses :"
MSG_COURSES_EMPTY = "ℹ️ Aucun ingrédient sélectionné."
MSG_COURSES_TIMEOUT = "⏰ Temps écoulé. Voici votre liste partielle :"

POLL_TIMEOUT_SECONDS = 300

# Conversation prompts — sent when a command is invoked without its required argument.
MSG_AJOUTER_PROMPT = "🔗 Envoyez-moi l'URL de la recette :"
MSG_CHERCHER_PROMPT = "🔍 Envoyez-moi le texte de recherche :"
MSG_PERSONNES_PROMPT = "👥 Envoyez-moi le nombre de personnes :"
MSG_ANNULER = "❌ Commande annulée."

# ConversationHandler states
STATE_AWAITING_URL = 0
STATE_AWAITING_QUERY = 1
STATE_AWAITING_SERVINGS = 2

# Global error handler (catch-all)
MSG_GENERIC_ERROR = "❌ Une erreur inattendue est survenue."
MSG_NETWORK_ERROR = "❌ Erreur réseau. Réessayez plus tard."

# Error categorisation table — maps exception types to (category, user-facing
# message, log severity). Checked via isinstance in first-match order; put
# subclasses before their base classes.
_ERROR_CATEGORY_MAP: list[tuple[type[BaseException], str, str, int]] = [
    # Network / Telegram transport
    (TelegramNetworkError, "network", MSG_NETWORK_ERROR, logging.WARNING),
    # Agent (subclasses first)
    (
        RecipeExtractionContentError,
        "agent_content",
        MSG_AJOUTER_VALIDATION_ERROR,
        logging.WARNING,
    ),
    (
        RecipeExtractionValidationError,
        "agent_validation",
        MSG_AJOUTER_VALIDATION_ERROR,
        logging.ERROR,
    ),
    (RecipeExtractionTimeout, "agent_timeout", MSG_AJOUTER_AGENT_ERROR, logging.WARNING),
    (RecipeExtractionAgentError, "agent", MSG_AJOUTER_AGENT_ERROR, logging.ERROR),
    # Embedding
    (EmbeddingError, "embedding", MSG_CHERCHER_EMBEDDING_ERROR, logging.ERROR),
    # PDF
    (PdfGenerationError, "pdf", MSG_RECIPE_PDF_ERROR, logging.ERROR),
    # Database (SQLAlchemy catch-all)
    (SQLAlchemyError, "database", MSG_AJOUTER_DB_ERROR, logging.ERROR),
]

_LOGFIRE_BY_LEVEL: dict[int, Any] = {
    logging.WARNING: logfire.warn,
    logging.ERROR: logfire.error,
}


# --- helpers --------------------------------------------------------------


def _resolve_servings(recipe: Recipe, user_data: dict[str, Any] | None) -> int:
    """Return adjusted servings from *user_data*, falling back to the recipe's original."""
    if user_data is not None:
        return user_data.get(USER_DATA_ADJUSTED_SERVINGS, recipe.servings)
    return recipe.servings


async def _send_recipe_pdf(
    message: object,
    recipe: Recipe,
    ingredients: list[Ingredient],
    servings: int,
) -> bool:
    """Generate a PDF for *recipe* and send it via *message*.

    Returns ``True`` on success, ``False`` if PDF generation or sending failed
    (the appropriate error reply is already sent to the user).
    """
    with logfire.span(
        "pdf_delivery",
        recipe_id=recipe.id,
        servings=servings,
    ) as span:
        try:
            pdf_bytes = generate_recipe_pdf(recipe, ingredients, servings)
        except PdfGenerationError:
            logfire.error("pdf.generation_failed", recipe_id=recipe.id)
            await message.reply_text(MSG_RECIPE_PDF_ERROR)  # type: ignore[union-attr]
            span.set_attribute("success", False)
            return False

        try:
            await message.reply_document(  # type: ignore[union-attr]
                document=pdf_bytes,
                filename=f"{recipe.title}.pdf",
            )
        except Exception:
            logfire.error("pdf.send_failed", recipe_id=recipe.id)
            await message.reply_text(MSG_RECIPE_SEND_ERROR)  # type: ignore[union-attr]
            span.set_attribute("success", False)
            return False

        span.set_attribute("success", True)
        return True


# --- command handlers -----------------------------------------------------


async def ajouter_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle ``/ajouter [url]``.

    If a URL argument is provided, execute immediately. Otherwise prompt the
    user and transition to :data:`STATE_AWAITING_URL`.
    """
    args = context.args or []
    if args:
        await _ajouter_logic(update, context, url=args[0])
        return ConversationHandler.END

    message = update.effective_message
    if message is None:
        return ConversationHandler.END
    await message.reply_text(MSG_AJOUTER_PROMPT)
    return STATE_AWAITING_URL


async def _ajouter_receive_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receive the URL sent after ``/ajouter`` was invoked without arguments."""
    message = update.effective_message
    if message is None:
        return ConversationHandler.END
    url = (message.text or "").strip()
    await _ajouter_logic(update, context, url=url)
    return ConversationHandler.END


async def _ajouter_logic(update: Update, context: ContextTypes.DEFAULT_TYPE, *, url: str) -> None:
    """Core logic for ``/ajouter``: validate URL, extract recipe, persist."""
    user_id = update.effective_user.id if update.effective_user else 0

    with logfire.span("command.ajouter", telegram_user_id=user_id, url=url):
        message = update.effective_message
        if message is None:
            return

        if not url:
            await message.reply_text(MSG_AJOUTER_USAGE)
            return

        if not (url.startswith("http://") or url.startswith("https://")):
            await message.reply_text(MSG_AJOUTER_INVALID_URL)
            return

        await message.reply_text(MSG_AJOUTER_ACK)

        try:
            recipe_data = await extract_recipe(url)
        except RecipeExtractionContentError as exc:
            await message.reply_text(MSG_AJOUTER_CONTENT_ERROR.format(reason=exc))
            return
        except RecipeExtractionValidationError:
            await message.reply_text(MSG_AJOUTER_VALIDATION_ERROR)
            return
        except (RecipeExtractionTimeout, RecipeExtractionAgentError):
            await message.reply_text(MSG_AJOUTER_AGENT_ERROR)
            return

        try:
            title_embedding = await encode_text(recipe_data.title)
        except EmbeddingError:
            await message.reply_text(MSG_AJOUTER_DB_ERROR)
            return

        try:
            async with session_scope() as session:
                recipe = Recipe(
                    telegram_user_id=user_id,
                    source_url=url,
                    title=recipe_data.title,
                    servings=recipe_data.servings,
                    steps=json.dumps(recipe_data.steps, ensure_ascii=False),
                    embedding=title_embedding,
                    ingredients=[
                        Ingredient(
                            name=ing.name,
                            quantity=ing.quantity,
                            unit=ing.unit,
                        )
                        for ing in recipe_data.ingredients
                    ],
                )
                session.add(recipe)
        except Exception:
            await message.reply_text(MSG_AJOUTER_DB_ERROR)
            return

        await message.reply_text(
            MSG_AJOUTER_SUCCESS.format(
                title=recipe_data.title,
                servings=recipe_data.servings,
            )
        )


async def chercher_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle ``/chercher [texte]``.

    If a query is provided, execute immediately. Otherwise prompt the user and
    transition to :data:`STATE_AWAITING_QUERY`.
    """
    args = context.args or []
    query = " ".join(args).strip()
    if query:
        await _chercher_logic(update, context, query=query)
        return ConversationHandler.END

    message = update.effective_message
    if message is None:
        return ConversationHandler.END
    await message.reply_text(MSG_CHERCHER_PROMPT)
    return STATE_AWAITING_QUERY


async def _chercher_receive_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receive the search query sent after ``/chercher`` was invoked without arguments."""
    message = update.effective_message
    if message is None:
        return ConversationHandler.END
    query = (message.text or "").strip()
    await _chercher_logic(update, context, query=query)
    return ConversationHandler.END


async def _chercher_logic(
    update: Update, context: ContextTypes.DEFAULT_TYPE, *, query: str
) -> None:
    """Core logic for ``/chercher``: encode query, vector search, show results."""
    user_id = update.effective_user.id if update.effective_user else 0

    with logfire.span("command.chercher", telegram_user_id=user_id, query=query):
        message = update.effective_message
        if message is None:
            return

        if not query:
            await message.reply_text(MSG_CHERCHER_USAGE)
            return

        await message.reply_text(MSG_CHERCHER_ACK)

        try:
            query_vec = await encode_text(query)
        except EmbeddingError:
            await message.reply_text(MSG_CHERCHER_EMBEDDING_ERROR)
            return

        settings = get_settings()
        try:
            async with session_scope() as session:
                dist_expr = Recipe.embedding.cosine_distance(query_vec)
                stmt = (
                    select(Recipe)
                    .where(Recipe.telegram_user_id == user_id)
                    .where(Recipe.embedding.is_not(None))
                    .where(dist_expr <= settings.search_distance_threshold)
                    .order_by(dist_expr)
                    .limit(settings.search_result_limit)
                )
                results = (await session.exec(stmt)).all()
        except Exception:
            await message.reply_text(MSG_CHERCHER_DB_ERROR)
            return

        with logfire.span(
            "search.vector",
            telegram_user_id=user_id,
            query_text=query,
            result_count=len(results),
        ):
            if not results:
                await message.reply_text(
                    MSG_CHERCHER_NO_RESULTS.format(query=query),
                )
                return

            keyboard = [
                [
                    InlineKeyboardButton(
                        text=f"{recipe.title} ({recipe.servings} pers.)",
                        callback_data=str(recipe.id),
                    )
                ]
                for recipe in results
            ]
            await message.reply_text(
                "📋 Résultats :",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )


async def recipe_selected_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline-keyboard tap from ``/chercher`` results.

    Fetches the selected recipe, generates a PDF (with optional serving
    adjustment from ``/personnes``), and sends it as a Telegram document.
    """
    callback_query = update.callback_query
    if callback_query is None:
        return

    user_id = update.effective_user.id if update.effective_user else 0

    with logfire.span(
        "callback.recipe_selected",
        telegram_user_id=user_id,
        callback_data=callback_query.data,
    ):
        await callback_query.answer()

        try:
            recipe_id = int(callback_query.data)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return

        if context.user_data is not None:
            context.user_data[USER_DATA_SELECTED_RECIPE_ID] = recipe_id

        if callback_query.message is None:
            return

        # Fetch recipe and generate PDF inside the session so
        # selectin-loaded ingredients remain accessible.
        async with session_scope() as session:
            recipe = await session.get(Recipe, recipe_id)
            if recipe is None:
                await callback_query.message.reply_text(MSG_RECIPE_NOT_FOUND)
                return

            servings = _resolve_servings(recipe, context.user_data)
            pdf_result = await _send_recipe_pdf(
                callback_query.message, recipe, recipe.ingredients, servings
            )

        # Clear adjusted servings after successful delivery.
        if pdf_result and context.user_data is not None:
            context.user_data.pop(USER_DATA_ADJUSTED_SERVINGS, None)


async def personnes_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle ``/personnes [n]``.

    If a number is provided, execute immediately. Otherwise prompt the user and
    transition to :data:`STATE_AWAITING_SERVINGS`.
    """
    args = context.args or []
    if args:
        await _personnes_logic(update, context, raw=args[0])
        return ConversationHandler.END

    message = update.effective_message
    if message is None:
        return ConversationHandler.END
    await message.reply_text(MSG_PERSONNES_PROMPT)
    return STATE_AWAITING_SERVINGS


async def _personnes_receive_servings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receive the servings count sent after ``/personnes`` was invoked without arguments."""
    message = update.effective_message
    if message is None:
        return ConversationHandler.END
    raw = (message.text or "").strip()
    await _personnes_logic(update, context, raw=raw)
    return ConversationHandler.END


async def _personnes_logic(update: Update, context: ContextTypes.DEFAULT_TYPE, *, raw: str) -> None:
    """Core logic for ``/personnes``: validate, store servings, generate PDF."""
    user_id = update.effective_user.id if update.effective_user else 0

    try:
        parsed: int | None = int(raw) if raw else None
    except ValueError:
        parsed = None

    with logfire.span("command.personnes", telegram_user_id=user_id, n=parsed):
        message = update.effective_message
        if message is None:
            return

        if not raw:
            await message.reply_text(MSG_PERSONNES_USAGE)
            return

        if parsed is None or parsed <= 0:
            await message.reply_text(MSG_PERSONNES_INVALID)
            return

        if context.user_data is None or not context.user_data.get(USER_DATA_SELECTED_RECIPE_ID):
            await message.reply_text(MSG_PERSONNES_NO_RECIPE)
            return

        recipe_id = context.user_data[USER_DATA_SELECTED_RECIPE_ID]
        async with session_scope() as session:
            recipe = await session.get(Recipe, recipe_id)
            if recipe is None:
                await message.reply_text(MSG_RECIPE_NOT_FOUND)
                return

            context.user_data[USER_DATA_ADJUSTED_SERVINGS] = parsed
            await message.reply_text(
                MSG_PERSONNES_ACK.format(n=parsed, original_servings=recipe.servings)
            )

            await _send_recipe_pdf(message, recipe, recipe.ingredients, parsed)


async def annuler_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle ``/annuler`` — abort the current conversation prompt."""
    message = update.effective_message
    if message is not None:
        await message.reply_text(MSG_ANNULER)
    return ConversationHandler.END


def _build_shopping_list(
    selected: list[Ingredient],
    servings: int,
    original_servings: int,
    *,
    timed_out: bool = False,
) -> str:
    """Build the full shopping-list message text."""
    if not selected:
        if timed_out:
            return MSG_COURSES_TIMEOUT
        return MSG_COURSES_EMPTY

    lines = [MSG_COURSES_HEADER]
    for ing in selected:
        line = format_ingredient_line(ing.name, ing.quantity, ing.unit, servings, original_servings)
        lines.append(f"- {line}")
    text = "\n".join(lines)

    if timed_out:
        return MSG_COURSES_TIMEOUT + "\n" + text
    return text


def _cleanup_poll_group(group: dict[str, Any], bot_data: dict[str, Any]) -> None:
    """Remove all poll-id entries for *group* from *bot_data*."""
    polls = bot_data.get("polls", {})
    for pid in group["poll_ids"]:
        polls.pop(pid, None)


async def _poll_timeout(bot: Bot, group: dict[str, Any], bot_data: dict[str, Any]) -> None:
    """Send the shopping list after :data:`POLL_TIMEOUT_SECONDS` if still pending."""
    await asyncio.sleep(POLL_TIMEOUT_SECONDS)
    if group["done"]:
        return
    group["done"] = True
    selected = [group["ingredients"][i] for i in sorted(group["selected_indices"])]
    text = _build_shopping_list(
        selected, group["servings"], group["original_servings"], timed_out=True
    )
    with logfire.span(
        "courses.timeout",
        chat_id=group["chat_id"],
        selected_count=len(selected),
    ):
        await bot.send_message(chat_id=group["chat_id"], text=text)
    group["timeout_task"] = None
    _cleanup_poll_group(group, bot_data)


async def courses_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle ``/courses``.

    Fetches the selected recipe, sends one or more ingredient polls (max 10
    options each), and schedules a 5-minute timeout. The companion
    :func:`poll_answer_handler` collects answers and produces the shopping list.
    """
    user_id = update.effective_user.id if update.effective_user else 0
    user_data = context.user_data
    recipe_id = user_data.get(USER_DATA_SELECTED_RECIPE_ID) if user_data else None

    with logfire.span("command.courses", telegram_user_id=user_id, recipe_id=recipe_id):
        message = update.effective_message
        if message is None:
            return ConversationHandler.END

        if not recipe_id:
            await message.reply_text(MSG_COURSES_NO_RECIPE)
            return ConversationHandler.END

        await message.reply_text(MSG_COURSES_ACK)

        try:
            async with session_scope() as session:
                recipe = await session.get(Recipe, recipe_id)
                # Materialise ingredients while the session is open so
                # selectin-loaded relations are accessible after exit.
                ingredients: list[Ingredient] = list(recipe.ingredients) if recipe else []
        except Exception:
            await message.reply_text(MSG_COURSES_DB_ERROR)
            return ConversationHandler.END

        if recipe is None:
            await message.reply_text(MSG_RECIPE_NOT_FOUND)
            return ConversationHandler.END
        if not ingredients:
            await message.reply_text(MSG_COURSES_NO_INGREDIENTS)
            return ConversationHandler.END

        servings = _resolve_servings(recipe, user_data)

        # Split ingredients into poll chunks of 10.
        chunks: list[list[Ingredient]] = [
            ingredients[i : i + 10] for i in range(0, len(ingredients), 10)
        ]
        multi = len(chunks) > 1

        # Shared mutable state for all polls in this group.
        group: dict[str, Any] = {
            "chat_id": message.chat.id,
            "ingredients": ingredients,
            "servings": servings,
            "original_servings": recipe.servings,
            "poll_ids": [],
            "option_maps": {},
            "selected_indices": set(),
            "answered_polls": set(),
            "timeout_task": None,
            "done": False,
        }

        offset = 0
        for chunk in chunks:
            options = [ing.name[:100] for ing in chunk]
            if multi:
                question = MSG_COURSES_POLL_QUESTION_MULTI.format(
                    start=offset + 1, end=offset + len(chunk)
                )
            else:
                question = MSG_COURSES_POLL_QUESTION

            poll_msg = await message.reply_poll(
                question=question,
                options=options,
                allows_multiple_answers=True,
                is_anonymous=False,
            )

            poll_id = poll_msg.poll.id
            group["poll_ids"].append(poll_id)
            group["option_maps"][poll_id] = {
                opt_idx: offset + opt_idx for opt_idx in range(len(chunk))
            }

            offset += len(chunk)

        # Register every poll_id → same group object.
        if "polls" not in context.bot_data:
            context.bot_data["polls"] = {}
        for pid in group["poll_ids"]:
            context.bot_data["polls"][pid] = group

        group["timeout_task"] = asyncio.create_task(
            _poll_timeout(context.bot, group, context.bot_data)
        )

        logfire.info(
            "courses.polls_sent",
            recipe_id=recipe_id,
            ingredient_count=len(ingredients),
            poll_count=len(chunks),
        )
        return ConversationHandler.END


async def poll_answer_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Collect poll answers and send the shopping list when all polls are answered."""
    poll_answer = update.poll_answer
    if poll_answer is None:
        return

    poll_id = poll_answer.poll_id
    polls = context.bot_data.get("polls", {})
    group = polls.get(poll_id)
    if group is None or group["done"]:
        return

    with logfire.span("courses.poll_answer", poll_id=poll_id):
        option_map = group["option_maps"].get(poll_id, {})
        # Remove any previous selection for this poll before applying the new one,
        # so deselecting an option in Telegram is reflected correctly.
        all_indices_for_poll = set(option_map.values())
        group["selected_indices"] -= all_indices_for_poll
        for opt_idx in poll_answer.option_ids:
            global_idx = option_map.get(opt_idx)
            if global_idx is not None:
                group["selected_indices"].add(global_idx)

        group["answered_polls"].add(poll_id)

        if len(group["answered_polls"]) < len(group["poll_ids"]):
            return

        # All polls answered — send the shopping list.
        group["done"] = True
        timeout_task = group.get("timeout_task")
        if timeout_task is not None:
            timeout_task.cancel()
        group["timeout_task"] = None

        selected = [group["ingredients"][i] for i in sorted(group["selected_indices"])]
        text = _build_shopping_list(selected, group["servings"], group["original_servings"])

        try:
            with logfire.span(
                "courses.shopping_list",
                selected_count=len(selected),
                servings=group["servings"],
            ):
                await context.bot.send_message(chat_id=group["chat_id"], text=text)
        finally:
            _cleanup_poll_group(group, context.bot_data)


# --- error handler --------------------------------------------------------


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler — categorise, log at the right severity, reply.

    Implements the categorisation table (network / agent / embedding / pdf /
    database / unknown). Each category is mapped to an appropriate French user
    message and log severity via ``_ERROR_CATEGORY_MAP``. Per-command handlers
    already catch *most* known exceptions with context-specific messages; this
    handler is the safety net for anything that escapes them.
    """
    error = context.error

    # 1. Categorise via first-match isinstance.
    category = "unknown"
    user_msg = MSG_GENERIC_ERROR
    level = logging.ERROR

    for exc_type, cat, msg, lvl in _ERROR_CATEGORY_MAP:
        if isinstance(error, exc_type):
            category, user_msg, level = cat, msg, lvl
            break

    # 2. Log with the appropriate severity and structured attributes.
    log_fn = _LOGFIRE_BY_LEVEL.get(level, logfire.error)
    log_fn(
        "bot.error.{category}",
        category=category,
        error_type=type(error).__name__ if error is not None else "None",
        error_message=str(error) if error is not None else "",
    )

    if not isinstance(update, Update):
        # Job-queue or background-task error: nothing to reply to.
        return

    message = update.effective_message
    if message is None:
        return

    # Reply itself can fail (network down, user blocked the bot, …). Swallow
    # so a failed error reply does not recurse back into this handler.
    with contextlib.suppress(Exception):
        await message.reply_text(user_msg)


# --- access control gate ---------------------------------------------------


async def _access_control_gate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Block updates from unauthorized users/chats.

    Registered in handler group ``-1`` so it runs *before* every other
    handler.  When the update originates from an ID that is not present in
    any of the configured allow-lists, the function raises
    :class:`ApplicationHandlerStop` to silently abort further processing.
    """
    settings = get_settings()
    if not settings.has_access_control:
        return

    user_id = update.effective_user.id if update.effective_user else None
    chat_id = update.effective_chat.id if update.effective_chat else None

    if user_id is not None and user_id in settings.allowed_user_id_set:
        return
    if chat_id is not None and chat_id in settings.allowed_group_id_set:
        return
    if chat_id is not None and chat_id in settings.allowed_channel_id_set:
        return

    logger.debug("Access denied for user_id=%s chat_id=%s", user_id, chat_id)
    raise ApplicationHandlerStop()


# --- application factory and entry point ----------------------------------


def build_application() -> Application:
    """Build the python-telegram-bot ``Application`` with all handlers registered.

    Separated from :func:`run` so unit tests can introspect the registered
    handlers without spawning a network loop.
    """
    settings = get_settings()
    app = Application.builder().token(settings.telegram_bot_token).build()

    # Access-control gate — runs before every other handler (group -1).
    # Only registered when at least one allow-list is non-empty so there is
    # zero overhead in the default open-access configuration.
    if settings.has_access_control:
        app.add_handler(TypeHandler(Update, _access_control_gate), group=-1)

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("ajouter", ajouter_command),
            CommandHandler("chercher", chercher_command),
            CommandHandler("personnes", personnes_command),
        ],
        states={
            STATE_AWAITING_URL: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, _ajouter_receive_url),
            ],
            STATE_AWAITING_QUERY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, _chercher_receive_query),
            ],
            STATE_AWAITING_SERVINGS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, _personnes_receive_servings),
            ],
        },
        fallbacks=[
            CommandHandler("annuler", annuler_command),
            CommandHandler("ajouter", ajouter_command),
            CommandHandler("chercher", chercher_command),
            CommandHandler("personnes", personnes_command),
            CommandHandler("courses", courses_command),
        ],
    )
    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("courses", courses_command))
    app.add_handler(CallbackQueryHandler(recipe_selected_callback))
    app.add_handler(PollAnswerHandler(poll_answer_handler))

    app.add_error_handler(error_handler)
    return app


def run() -> None:
    """Build the application and start polling. Blocks until SIGINT/SIGTERM."""
    # Pre-load the sentence-transformers model once, synchronously, before
    # the event loop starts. This pays the ~400 MB cold-start cost up front
    # so the first /ajouter or /chercher does not stall on model download
    # inside an update handler. Kept out of build_application() so unit tests
    # can introspect handlers without triggering a real model download.
    get_embedding_model()

    # Initialize Logfire — if no token is set, spans remain no-ops
    settings = get_settings()
    if settings.logfire_token:
        logfire.configure(token=settings.logfire_token)

    app = build_application()
    app.run_polling(drop_pending_updates=True)
