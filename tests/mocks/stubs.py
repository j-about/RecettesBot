"""Plug-in stubs for agent extraction and embedding encoding.

Each factory returns an async callable that can replace the real function
via ``monkeypatch.setattr(bot, "extract_recipe", stub)``. The stub records
every call and either returns a preset result or raises a preset exception.
"""

from agent import RecipeOutput
from tests.mocks.factories import sample_recipe_output


def stub_extract_recipe(
    result: RecipeOutput | None = None,
    *,
    raise_exc: BaseException | None = None,
):
    """Return an async callable that replaces ``bot.extract_recipe``.

    *result* defaults to :func:`sample_recipe_output` when not provided
    and *raise_exc* is ``None``.
    """
    calls: list[str] = []

    async def _fake(url: str) -> RecipeOutput:
        calls.append(url)
        if raise_exc is not None:
            raise raise_exc
        return result if result is not None else sample_recipe_output()

    _fake.calls = calls  # type: ignore[attr-defined]
    return _fake


def stub_encode_text(
    result: list[float] | None = None,
    *,
    raise_exc: BaseException | None = None,
):
    """Return an async callable that replaces ``bot.encode_text``.

    Returns a deterministic 768-dim zero vector by default.
    """
    calls: list[str] = []

    async def _fake(text: str) -> list[float]:
        calls.append(text)
        if raise_exc is not None:
            raise raise_exc
        return result if result is not None else [0.0] * 768

    _fake.calls = calls  # type: ignore[attr-defined]
    return _fake
