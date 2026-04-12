"""Unit tests for ``embedding``.

Every test stubs ``sentence_transformers.SentenceTransformer`` via monkeypatch
— no real model is downloaded, no HuggingFace network call. The fake records
its constructor and ``encode`` calls so each branch of the embedding service
can be exercised: 768-dim list output, thread offload, model load failure,
encode failure, lru_cache singleton, and the Logfire span attributes.

Mirrors the pattern in :mod:`tests.unit.test_agent`, which patches
``agent.ResultMessage`` to a dataclass fake for the same reason.
"""

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

import embedding
from embedding import (
    EmbeddingEncodeError,
    EmbeddingModelLoadError,
    encode_text,
    get_embedding_model,
)

# --- fakes ----------------------------------------------------------------


@dataclass
class _FakeSentenceTransformer:
    """Stand-in for ``sentence_transformers.SentenceTransformer``.

    Records the model name passed to the constructor and every text passed
    to ``encode``. Returns a deterministic 768-dim ``float32`` array so the
    tests can assert on shape and dtype without depending on the real model.
    """

    name: str
    encoded: list[str] = field(default_factory=list)
    encode_thread_ids: list[int] = field(default_factory=list)
    sleep_seconds: float = 0.0
    raise_on_encode: BaseException | None = None

    def encode(self, text: str, **kwargs: Any) -> np.ndarray:
        if self.sleep_seconds:
            import time as _time

            _time.sleep(self.sleep_seconds)
        self.encoded.append(text)
        self.encode_thread_ids.append(threading.get_ident())
        if self.raise_on_encode is not None:
            raise self.raise_on_encode
        # 768 floats, distinct values so .tolist() can't accidentally yield
        # a list of zeros that masks a bug.
        return np.arange(768, dtype=np.float32) / 768.0


def _patch_constructor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    factory=None,
    raise_on_init: BaseException | None = None,
) -> dict[str, Any]:
    """Patch ``embedding.SentenceTransformer`` with a configurable factory.

    Returns a dict tracking ``call_count`` and the most recent ``instance``
    so tests can assert on cache behaviour and inspect the fake.
    """
    state: dict[str, Any] = {"call_count": 0, "instance": None}

    def _ctor(name: str, *args: Any, **kwargs: Any) -> _FakeSentenceTransformer:
        state["call_count"] += 1
        if raise_on_init is not None:
            raise raise_on_init
        instance = (factory or _FakeSentenceTransformer)(name=name)
        state["instance"] = instance
        return instance

    monkeypatch.setattr(embedding, "SentenceTransformer", _ctor)
    return state


# --- fixtures -------------------------------------------------------------


@pytest.fixture(autouse=True)
def embedding_env():
    """Reset the embedding-model singleton on entry and exit.

    Settings env vars and the ``get_settings`` cache are handled by the
    autouse ``_base_env`` fixture in tests/unit/conftest.py. This fixture
    only resets the model cache so a stub installed by one test cannot leak
    into a sibling test.
    """
    get_embedding_model.cache_clear()
    yield
    get_embedding_model.cache_clear()


# --- get_embedding_model() ------------------------------------------------


def test_get_embedding_model_returns_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _patch_constructor(monkeypatch)

    model = get_embedding_model()

    assert isinstance(model, _FakeSentenceTransformer)
    assert model.name == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    assert state["call_count"] == 1


def test_get_embedding_model_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _patch_constructor(monkeypatch)

    first = get_embedding_model()
    second = get_embedding_model()

    assert first is second
    assert state["call_count"] == 1, "constructor must run once thanks to lru_cache"


def test_get_embedding_model_wraps_load_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_constructor(monkeypatch, raise_on_init=OSError("model not found on disk"))

    with pytest.raises(EmbeddingModelLoadError, match="failed to load embedding model"):
        get_embedding_model()


def test_get_embedding_model_load_error_chains_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RuntimeError("CUDA out of memory")
    _patch_constructor(monkeypatch, raise_on_init=original)

    with pytest.raises(EmbeddingModelLoadError) as info:
        get_embedding_model()

    assert info.value.__cause__ is original


# --- encode_text() --------------------------------------------------------


async def test_encode_text_returns_768_dim_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_constructor(monkeypatch)

    result = await encode_text("Tarte aux pommes")

    assert isinstance(result, list)
    assert len(result) == 768
    assert all(isinstance(x, float) for x in result)


async def test_encode_text_passes_input_to_model(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _patch_constructor(monkeypatch)

    await encode_text("Bœuf bourguignon")

    assert state["instance"].encoded == ["Bœuf bourguignon"]


async def test_encode_text_runs_in_worker_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Asserts ``asyncio.to_thread`` is actually used (not a direct call)."""
    state = _patch_constructor(monkeypatch)

    await encode_text("crêpes")

    main_thread_id = threading.get_ident()
    [encode_thread_id] = state["instance"].encode_thread_ids
    assert encode_thread_id != main_thread_id


async def test_encode_text_handles_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty strings are valid input (users may send very short queries)."""
    _patch_constructor(monkeypatch)

    result = await encode_text("")

    assert len(result) == 768


async def test_encode_text_wraps_encode_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    boom = RuntimeError("model exploded")

    def factory(name: str) -> _FakeSentenceTransformer:
        return _FakeSentenceTransformer(name=name, raise_on_encode=boom)

    _patch_constructor(monkeypatch, factory=factory)

    with pytest.raises(EmbeddingEncodeError, match="failed to encode text of length 11") as info:
        await encode_text("ratatouille")

    assert info.value.__cause__ is boom


async def test_encode_text_completes_under_one_second(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding a single title must complete within 1 s on CPU.

    Smoke check with a 10 ms fake encode — exercises the timing path without
    paying for a real model.
    """
    import time

    def factory(name: str) -> _FakeSentenceTransformer:
        return _FakeSentenceTransformer(name=name, sleep_seconds=0.01)

    _patch_constructor(monkeypatch, factory=factory)

    start = time.perf_counter()
    await encode_text("salade niçoise")
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0


async def test_encode_text_opens_logfire_span(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture the span name + attributes via a fake ``logfire.span``.

    ``logfire.span`` is a no-op without ``logfire.configure``, so we replace
    it with a recording context manager. Mirrors how test_agent.py patches
    third-party symbols on the imported module.
    """
    recorded: dict[str, Any] = {"name": None, "init_attrs": {}, "set_attrs": {}}

    class _FakeSpan:
        def __enter__(self) -> "_FakeSpan":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def set_attribute(self, key: str, value: Any) -> None:
            recorded["set_attrs"][key] = value

    def fake_span(name: str, **attrs: Any) -> _FakeSpan:
        recorded["name"] = name
        recorded["init_attrs"] = attrs
        return _FakeSpan()

    monkeypatch.setattr(embedding.logfire, "span", fake_span)
    _patch_constructor(monkeypatch)

    await encode_text("crème brûlée")

    assert recorded["name"] == "embedding.encode"
    assert recorded["init_attrs"] == {"text_length": len("crème brûlée")}
    assert "duration_ms" in recorded["set_attrs"]
    assert isinstance(recorded["set_attrs"]["duration_ms"], float)
    assert recorded["set_attrs"]["duration_ms"] >= 0
