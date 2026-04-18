"""Shared fixtures for unit tests.

The ``_base_env`` autouse fixture provides the minimum Settings env vars so
every unit test starts from a known baseline and a clean ``get_settings``
cache. It also strips optional Settings env vars the developer may have in
their local shell or ``.env`` file (access-control allow-lists, Logfire
token, etc.) and disables ``.env`` loading entirely — unit tests must be
hermetic and never depend on the developer's machine. Modules that need
extra env vars (e.g. ``AGENT_TIMEOUT_SECONDS``) add them in their own
fixture and call ``get_settings.cache_clear()`` once more.

This fixture intentionally lives here (not in the root ``tests/conftest.py``)
because integration tests need real Postgres credentials from ``.env``, and an
autouse fixture that overwrites them would break those tests.
"""

import pytest

from settings import Settings, get_settings

_OPTIONAL_SETTINGS_ENV_VARS = (
    "ALLOWED_USER_IDS",
    "ALLOWED_GROUP_IDS",
    "EMBEDDING_MODEL_NAME",
    "SEARCH_DISTANCE_THRESHOLD",
    "SEARCH_RESULT_LIMIT",
    "AGENT_TIMEOUT_SECONDS",
    "LOGFIRE_TOKEN",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture(autouse=True)
def _base_env(monkeypatch: pytest.MonkeyPatch):
    for key in _OPTIONAL_SETTINGS_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(Settings, "model_config", {"env_file": None})
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-123")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x:x@db:5432/y")
    monkeypatch.setenv("DATABASE_URL_SYNC", "postgresql+psycopg://x:x@db:5432/y")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
