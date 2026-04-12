"""Dataclass fakes for Telegram API objects.

These satisfy the duck-typed attribute contract that the bot handlers read,
without pulling in the real python-telegram-bot network machinery. The fakes
record every ``reply_text``, ``reply_document``, ``reply_poll``, and
``send_message`` call so tests can assert on the exact message sequence.
"""

from dataclasses import dataclass, field
from typing import Any

_poll_counter: int = 0


@dataclass
class FakeChat:
    """Stand-in for ``telegram.Chat`` — only ``id`` is exercised."""

    id: int = 42


@dataclass
class FakePoll:
    """Stand-in for ``telegram.Poll`` — only ``id`` is exercised."""

    id: str = "poll_1"


@dataclass
class FakePollMessage:
    """Stand-in for the ``Message`` returned by ``reply_poll``."""

    poll: FakePoll = field(default_factory=FakePoll)


@dataclass
class FakeMessage:
    """Stand-in for ``telegram.Message``."""

    chat: FakeChat = field(default_factory=FakeChat)
    text: str | None = None
    replies: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    documents: list[dict[str, Any]] = field(default_factory=list)
    polls: list[dict[str, Any]] = field(default_factory=list)
    raise_on_reply: BaseException | None = None
    raise_on_document: BaseException | None = None

    async def reply_text(self, text: str, **kwargs: Any) -> None:
        if self.raise_on_reply is not None:
            raise self.raise_on_reply
        self.replies.append((text, kwargs))

    async def reply_document(self, document: Any, **kwargs: Any) -> None:
        if self.raise_on_document is not None:
            raise self.raise_on_document
        self.documents.append({"document": document, **kwargs})

    async def reply_poll(self, question: str, options: list[str], **kwargs: Any) -> FakePollMessage:
        global _poll_counter  # noqa: PLW0603
        _poll_counter += 1
        poll_id = f"poll_{_poll_counter}"
        self.polls.append({"question": question, "options": options, **kwargs})
        return FakePollMessage(poll=FakePoll(id=poll_id))


@dataclass
class FakeUser:
    """Stand-in for ``telegram.User`` — only ``id`` is exercised."""

    id: int = 42


@dataclass
class FakeCallbackQuery:
    """Stand-in for ``telegram.CallbackQuery``."""

    data: str | None = None
    message: FakeMessage | None = field(default_factory=FakeMessage)
    answered: bool = False

    async def answer(self, **kwargs: Any) -> bool:
        self.answered = True
        return True


@dataclass
class FakePollAnswer:
    """Stand-in for ``telegram.PollAnswer``."""

    poll_id: str = ""
    option_ids: list[int] = field(default_factory=list)
    user: FakeUser = field(default_factory=FakeUser)


@dataclass
class FakeUpdate:
    """Stand-in for ``telegram.Update``.

    The ``patch_update`` fixture rebinds ``bot.Update`` to this class so
    ``isinstance(update, Update)`` in ``bot.error_handler`` accepts it.
    """

    effective_user: FakeUser | None = field(default_factory=FakeUser)
    effective_message: FakeMessage | None = field(default_factory=FakeMessage)
    effective_chat: FakeChat | None = field(default_factory=FakeChat)
    callback_query: FakeCallbackQuery | None = None
    poll_answer: FakePollAnswer | None = None


@dataclass
class FakeBot:
    """Stand-in for ``telegram.Bot`` — records ``send_message`` calls."""

    sent_messages: list[dict[str, Any]] = field(default_factory=list)

    async def send_message(self, chat_id: int, text: str, **kwargs: Any) -> None:
        self.sent_messages.append({"chat_id": chat_id, "text": text, **kwargs})


@dataclass
class FakeContext:
    """Stand-in for ``telegram.ext.CallbackContext``."""

    args: list[str] = field(default_factory=list)
    user_data: dict[str, Any] | None = field(default_factory=dict)
    bot_data: dict[str, Any] = field(default_factory=dict)
    bot: FakeBot = field(default_factory=FakeBot)
    error: BaseException | None = None
