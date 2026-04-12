"""Shared mock objects and factory functions for RecettesBot tests.

This package provides reusable fakes, stubs, and factory functions for all
external dependencies — designed to be imported by both unit and e2e tests.

Modules
-------
telegram : Dataclass fakes for Telegram API objects (Update, Message, etc.)
database : Fake session implementations and session-scope replacements
factories : Builder functions for Recipe, Ingredient, and RecipeOutput data
stubs : Plug-in stubs for agent extraction and embedding encoding
"""

from tests.mocks.database import (
    FakeGetSession,
    FakeSearchResult,
    FakeSearchSession,
    FakeSession,
    failing_session_scope,
    fake_session_scope,
)
from tests.mocks.factories import (
    make_fake_ingredients,
    make_fake_recipe,
    sample_recipe_output,
)
from tests.mocks.stubs import (
    stub_encode_text,
    stub_extract_recipe,
)
from tests.mocks.telegram import (
    FakeBot,
    FakeCallbackQuery,
    FakeChat,
    FakeContext,
    FakeMessage,
    FakePoll,
    FakePollAnswer,
    FakePollMessage,
    FakeUpdate,
    FakeUser,
)

__all__ = [
    "FakeBot",
    "FakeCallbackQuery",
    "FakeChat",
    "FakeContext",
    "FakeGetSession",
    "FakeMessage",
    "FakePoll",
    "FakePollAnswer",
    "FakePollMessage",
    "FakeSearchResult",
    "FakeSearchSession",
    "FakeSession",
    "FakeUpdate",
    "FakeUser",
    "failing_session_scope",
    "fake_session_scope",
    "make_fake_ingredients",
    "make_fake_recipe",
    "sample_recipe_output",
    "stub_encode_text",
    "stub_extract_recipe",
]
