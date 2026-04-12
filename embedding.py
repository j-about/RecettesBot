"""Sentence-transformers embedding service for recipe title vectors.

Exposes :func:`encode_text`, a coroutine that turns a French recipe title or
search query into a 768-dimensional ``list[float]`` suitable for the
``recipes.embedding`` column (:class:`models.Recipe`).

Lifecycle
---------
The SentenceTransformer model is loaded lazily by :func:`get_embedding_model`
and cached for the lifetime of the process. The runtime entry point,
:func:`bot.run`, calls ``get_embedding_model()`` once before polling starts so
the first ``/ajouter`` does not pay the ~400 MB cold-start cost inside a
Telegram update handler.

Threading
---------
``SentenceTransformer.encode`` is CPU-bound and synchronous. A single title
encodes in 200-600 ms on CPU with the pinned model. :func:`encode_text`
wraps the call in ``asyncio.to_thread`` so the event loop keeps handling other
updates while the model runs its forward pass.

Return type
-----------
``encode_text`` returns ``list[float]`` (not ``numpy.ndarray``) to match
:attr:`models.Recipe.embedding` and to keep numpy out of the call sites in
``bot.py``. pgvector's SQLAlchemy bind processor expects the text form
anyway — see ``db.py`` for the rationale.
"""

import asyncio
import time
from functools import lru_cache

import logfire
from sentence_transformers import SentenceTransformer

from settings import get_settings

__all__ = [
    "EmbeddingEncodeError",
    "EmbeddingError",
    "EmbeddingModelLoadError",
    "encode_text",
    "get_embedding_model",
]


class EmbeddingError(Exception):
    """Base class for any failure inside the embedding service.

    The global error handler catches this and maps it to the French
    "embedding error" user message. Catch the base class to handle "any
    embedding failure"; catch a concrete subclass to distinguish startup
    vs. runtime faults.
    """


class EmbeddingModelLoadError(EmbeddingError):
    """The sentence-transformers model failed to load from disk / HuggingFace."""


class EmbeddingEncodeError(EmbeddingError):
    """A runtime ``encode`` call raised during text encoding."""


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Return the singleton :class:`SentenceTransformer`, loading on first call.

    Wrapped in ``lru_cache`` for the same reason ``settings.get_settings`` and
    ``db.get_engine`` are: one instance per process, no module-level side
    effects at import time, easy ``cache_clear()`` in tests.

    Raises:
        EmbeddingModelLoadError: the model name configured in Settings could
            not be loaded (missing network, disk full, unknown model, OOM).
    """
    settings = get_settings()
    try:
        return SentenceTransformer(settings.embedding_model_name)
    except Exception as exc:
        # sentence-transformers / huggingface_hub raise a variety of concrete
        # types (OSError, RepositoryNotFoundError, RuntimeError, ...) — wrap
        # them all in a single domain exception so the error handler has one
        # class to catch.
        raise EmbeddingModelLoadError(
            f"failed to load embedding model {settings.embedding_model_name!r}"
        ) from exc


async def encode_text(text: str) -> list[float]:
    """Encode ``text`` into a 768-dimensional ``list[float]``.

    Wrapped in a Logfire span (``embedding.encode``) with ``text_length`` and
    ``duration_ms`` attributes. The CPU-bound ``model.encode`` call is
    offloaded to a worker thread via ``asyncio.to_thread`` to keep the event
    loop responsive.

    The return type matches :attr:`models.Recipe.embedding` exactly
    (``list[float] | None``), so callers can assign the result directly to a
    Recipe instance.

    Raises:
        EmbeddingEncodeError: the model raised during encoding. In practice
            this should not happen at runtime for valid UTF-8 input, but the
            exception exists so the global error handler has a concrete class
            to catch.
    """
    model = get_embedding_model()  # cached after the first call

    with logfire.span("embedding.encode", text_length=len(text)) as span:
        start = time.perf_counter()
        try:
            vector = await asyncio.to_thread(model.encode, text)
        except Exception as exc:
            raise EmbeddingEncodeError(f"failed to encode text of length {len(text)}") from exc
        duration_ms = (time.perf_counter() - start) * 1000
        span.set_attribute("duration_ms", duration_ms)

    # SentenceTransformer.encode() with default kwargs returns a numpy.ndarray;
    # .tolist() gives us the list[float] that pgvector's SQLAlchemy
    # bind_processor expects (see db.py for why we use the text path instead
    # of the asyncpg binary codec).
    return vector.tolist()
