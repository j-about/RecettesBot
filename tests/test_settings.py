import pytest
from pydantic import ValidationError

from settings import Settings, get_settings

REQUIRED_ENV = {
    "TELEGRAM_BOT_TOKEN": "test-telegram-token",
}

DATABASE_ENV = {
    "DATABASE_URL": "postgresql+asyncpg://testuser:testpass@dbhost:5432/testdb",
    "DATABASE_URL_SYNC": "postgresql+psycopg://testuser:testpass@dbhost:5432/testdb",
}

OPTIONAL_DEFAULTS = {
    "EMBEDDING_MODEL_NAME",
    "SEARCH_DISTANCE_THRESHOLD",
    "SEARCH_RESULT_LIMIT",
    "AGENT_TIMEOUT_SECONDS",
    "LOGFIRE_TOKEN",
    "ALLOWED_USER_IDS",
    "ALLOWED_GROUP_IDS",
}


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Strip every Settings-related env var so each test starts from a clean slate.

    Also disables `.env` file loading so a developer's local file cannot leak in.
    """
    all_keys = {
        *REQUIRED_ENV,
        *DATABASE_ENV,
        *OPTIONAL_DEFAULTS,
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
    }
    for key in all_keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(Settings, "model_config", {"env_file": None})
    get_settings.cache_clear()
    return monkeypatch


def test_settings_loads_from_env(isolated_env: pytest.MonkeyPatch) -> None:
    for key, value in {**REQUIRED_ENV, **DATABASE_ENV}.items():
        isolated_env.setenv(key, value)

    s = Settings()
    assert s.telegram_bot_token == REQUIRED_ENV["TELEGRAM_BOT_TOKEN"]
    assert s.database_url == "postgresql+asyncpg://testuser:testpass@dbhost:5432/testdb"
    assert s.database_url_sync == "postgresql+psycopg://testuser:testpass@dbhost:5432/testdb"
    # Verify default values for optional settings.
    assert s.embedding_model_name == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    assert s.search_distance_threshold == 0.65
    assert s.search_result_limit == 5
    assert s.agent_timeout_seconds == 120
    assert s.logfire_token == ""
    assert s.allowed_user_ids == ""
    assert s.allowed_group_ids == ""


def test_settings_missing_required_raises(isolated_env: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError) as exc:
        Settings()
    missing = {err["loc"][0] for err in exc.value.errors()}
    assert {"telegram_bot_token", "database_url", "database_url_sync"} <= missing


# --- access-control settings -----------------------------------------------


def _make_settings(mp: pytest.MonkeyPatch, **extra: str) -> Settings:
    """Create a ``Settings`` with required env vars and optional *extra* overrides."""
    for key, value in REQUIRED_ENV.items():
        mp.setenv(key, value)
    for key, value in extra.items():
        mp.setenv(key, value)
    for key, value in DATABASE_ENV.items():
        mp.setenv(key, value)
    return Settings()


def test_allowed_ids_default_empty(isolated_env: pytest.MonkeyPatch) -> None:
    s = _make_settings(isolated_env)
    assert s.allowed_user_ids == ""
    assert s.allowed_group_ids == ""
    assert s.allowed_user_id_set == frozenset()
    assert s.allowed_group_id_set == frozenset()


def test_allowed_ids_parse_comma_separated(isolated_env: pytest.MonkeyPatch) -> None:
    s = _make_settings(isolated_env, ALLOWED_USER_IDS="123,456,789")
    assert s.allowed_user_id_set == frozenset({123, 456, 789})


def test_allowed_ids_parse_single(isolated_env: pytest.MonkeyPatch) -> None:
    s = _make_settings(isolated_env, ALLOWED_USER_IDS="123")
    assert s.allowed_user_id_set == frozenset({123})


def test_allowed_ids_parse_with_whitespace(isolated_env: pytest.MonkeyPatch) -> None:
    s = _make_settings(isolated_env, ALLOWED_USER_IDS=" 123 , 456 ")
    assert s.allowed_user_id_set == frozenset({123, 456})


def test_allowed_ids_parse_empty_string(isolated_env: pytest.MonkeyPatch) -> None:
    s = _make_settings(isolated_env, ALLOWED_USER_IDS="")
    assert s.allowed_user_id_set == frozenset()


def test_allowed_ids_invalid_raises(isolated_env: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        _make_settings(isolated_env, ALLOWED_USER_IDS="abc")


def test_allowed_ids_negative_ids(isolated_env: pytest.MonkeyPatch) -> None:
    """Supergroup IDs in Telegram are negative (e.g. -1001234567890)."""
    s = _make_settings(isolated_env, ALLOWED_GROUP_IDS="-1001234567890,-1009876543210")
    assert s.allowed_group_id_set == frozenset({-1001234567890, -1009876543210})


def test_has_access_control_false_when_empty(isolated_env: pytest.MonkeyPatch) -> None:
    s = _make_settings(isolated_env)
    assert s.has_access_control is False


def test_has_access_control_true_when_user_ids_set(
    isolated_env: pytest.MonkeyPatch,
) -> None:
    s = _make_settings(isolated_env, ALLOWED_USER_IDS="42")
    assert s.has_access_control is True


def test_has_access_control_true_when_group_ids_set(
    isolated_env: pytest.MonkeyPatch,
) -> None:
    s = _make_settings(isolated_env, ALLOWED_GROUP_IDS="-100123")
    assert s.has_access_control is True
