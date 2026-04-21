"""Application settings loaded from environment variables and ``.env``.

Provides a :class:`Settings` Pydantic model that validates and exposes all
configuration — Telegram credentials, database URLs, embedding model name,
search tuning parameters, agent timeout, Logfire token, and optional
access-control allow-lists for users and groups. The singleton is accessed
via :func:`get_settings`, which caches the instance for the lifetime of the
process.
"""

from functools import cached_property, lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # ANTHROPIC_API_KEY and CLAUDE_CODE_OAUTH_TOKEN (both read directly by
        # the Claude SDK from os.environ) are documented in .env.example and
        # live in .env, so the .env loader sees them. Ignore unknown keys
        # instead of failing on every load.
        extra="ignore",
    )

    # Telegram
    telegram_bot_token: str

    # Database — full connection URLs
    database_url: str
    database_url_sync: str

    # Claude Agent SDK
    # The SDK reads ANTHROPIC_API_KEY and CLAUDE_CODE_OAUTH_TOKEN from the
    # environment automatically. No explicit fields needed here, but
    # documented for clarity.

    # Sentence Transformers
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    # Search
    search_distance_threshold: float = 0.65
    search_result_limit: int = 5

    # Agent
    agent_timeout_seconds: int = 120

    # Logfire
    logfire_token: str = ""

    # Access control — restrict the bot to specific Telegram users or groups.
    # Each variable accepts a comma-separated list of integer IDs.  When both
    # are empty the bot is open to everyone.
    # Stored as ``str`` because pydantic-settings attempts JSON decoding on
    # complex types *before* field validators run, which rejects plain
    # comma-separated values.
    allowed_user_ids: str = ""
    allowed_group_ids: str = ""

    @field_validator("allowed_user_ids", "allowed_group_ids")
    @classmethod
    def _validate_comma_separated_ids(cls, value: str) -> str:
        if not value.strip():
            return ""
        for item in value.split(","):
            item = item.strip()
            if item:
                int(item)  # raises ValueError for non-integer tokens
        return value

    @staticmethod
    def _parse_id_list(value: str) -> frozenset[int]:
        if not value.strip():
            return frozenset()
        return frozenset(int(item.strip()) for item in value.split(",") if item.strip())

    @cached_property
    def allowed_user_id_set(self) -> frozenset[int]:
        """Parsed set of allowed Telegram user IDs."""
        return self._parse_id_list(self.allowed_user_ids)

    @cached_property
    def allowed_group_id_set(self) -> frozenset[int]:
        """Parsed set of allowed Telegram group IDs."""
        return self._parse_id_list(self.allowed_group_ids)

    @property
    def has_access_control(self) -> bool:
        """Return ``True`` when at least one allow-list is non-empty."""
        return bool(
            self.allowed_user_ids.strip()
            or self.allowed_group_ids.strip()
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance, loading env vars on first call."""
    return Settings()
