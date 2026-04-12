"""Unit tests for ``pdf``.

fpdf2 is pure Python with no external dependencies, so tests construct
:class:`models.Recipe` and :class:`models.Ingredient` instances directly
and call :func:`pdf.generate_recipe_pdf` — no fakes are needed for the
PDF library itself.

fpdf2 compresses its content stream with FlateDecode, so raw byte
assertions (``b"400" in result``) would fail.  Content-bearing tests
therefore decompress the stream and search the extracted text.

Logfire span assertions and error-wrapping tests follow the same
monkeypatch pattern used in :mod:`tests.unit.test_embedding`.
"""

import json
import re
import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest

import pdf
from models import Ingredient, Recipe
from pdf import PdfGenerationError, generate_recipe_pdf

# --- helpers ----------------------------------------------------------------


def _make_recipe(
    *,
    title: str = "Crêpes",
    servings: int = 4,
    steps: list[str] | None = None,
    recipe_id: int = 1,
) -> Recipe:
    if steps is None:
        steps = ["Mélanger les ingrédients.", "Cuire dans une poêle."]
    return Recipe(
        id=recipe_id,
        telegram_user_id=42,
        source_url="https://example.com/crepes",
        title=title,
        servings=servings,
        steps=json.dumps(steps, ensure_ascii=False),
    )


def _make_ingredients(
    specs: list[tuple[str, float | None, str | None]] | None = None,
) -> list[Ingredient]:
    """Build a list of Ingredient instances from ``(name, quantity, unit)`` tuples."""
    if specs is None:
        specs = [("farine", 250.0, "g"), ("lait", 500.0, "ml"), ("oeufs", 3.0, None)]
    return [
        Ingredient(name=name, quantity=qty, unit=unit, recipe_id=1) for name, qty, unit in specs
    ]


def _extract_pdf_text(data: bytes) -> str:
    """Decompress FlateDecode streams in *data* and return concatenated text.

    This is a minimal extractor — good enough for asserting that specific
    strings appear in the rendered PDF without pulling in a full PDF parser.
    """
    chunks: list[str] = []
    for match in re.finditer(rb"stream\r?\n(.+?)\r?\nendstream", data, re.DOTALL):
        try:
            raw = zlib.decompress(match.group(1))
            # Extract text operands from PDF content stream (Tj and TJ operators)
            for text_match in re.finditer(rb"\(([^)]*)\)", raw):
                chunks.append(text_match.group(1).decode("latin-1", errors="replace"))
        except zlib.error:
            continue
    return "".join(chunks)


# --- basic output -----------------------------------------------------------


class TestBasicOutput:
    def test_returns_bytes(self) -> None:
        recipe = _make_recipe()
        result = generate_recipe_pdf(recipe, _make_ingredients(), servings=4)
        assert isinstance(result, bytes)

    def test_valid_pdf_header(self) -> None:
        recipe = _make_recipe()
        result = generate_recipe_pdf(recipe, _make_ingredients(), servings=4)
        assert result[:5] == b"%PDF-"

    def test_nonempty(self) -> None:
        recipe = _make_recipe()
        result = generate_recipe_pdf(recipe, _make_ingredients(), servings=4)
        assert len(result) > 0


# --- quantity adjustment ----------------------------------------------------


class TestQuantityAdjustment:
    def test_doubles_quantity(self) -> None:
        recipe = _make_recipe(servings=4)
        ings = _make_ingredients([("farine", 200.0, "g")])
        result = generate_recipe_pdf(recipe, ings, servings=8)
        text = _extract_pdf_text(result)
        assert "400" in text

    def test_none_quantity(self) -> None:
        recipe = _make_recipe(servings=4)
        ings = _make_ingredients([("sel", None, None)])
        result = generate_recipe_pdf(recipe, ings, servings=8)
        text = _extract_pdf_text(result)
        assert "sel" in text

    def test_none_unit(self) -> None:
        recipe = _make_recipe(servings=4)
        ings = _make_ingredients([("oeufs", 3.0, None)])
        result = generate_recipe_pdf(recipe, ings, servings=4)
        text = _extract_pdf_text(result)
        assert "oeufs" in text

    def test_ratio_formatting(self) -> None:
        """100 * 2/3 = 66.667 -> formatted as '66.67' (2 decimal places)."""
        recipe = _make_recipe(servings=3)
        ings = _make_ingredients([("farine", 100.0, "g")])
        result = generate_recipe_pdf(recipe, ings, servings=2)
        text = _extract_pdf_text(result)
        assert "66.67" in text

    def test_same_servings_no_change(self) -> None:
        recipe = _make_recipe(servings=4)
        ings = _make_ingredients([("farine", 250.0, "g")])
        result = generate_recipe_pdf(recipe, ings, servings=4)
        text = _extract_pdf_text(result)
        assert "250" in text


# --- steps ------------------------------------------------------------------


class TestSteps:
    def test_multiple_steps_numbered(self) -> None:
        steps = ["Faire la pâte.", "Laisser reposer.", "Cuire."]
        recipe = _make_recipe(steps=steps)
        result = generate_recipe_pdf(recipe, _make_ingredients(), servings=4)
        text = _extract_pdf_text(result)
        assert "1." in text
        assert "2." in text
        assert "3." in text


# --- french accents ---------------------------------------------------------


class TestFrenchAccents:
    def test_accented_title(self) -> None:
        recipe = _make_recipe(title="Crêpes bretonnes")
        result = generate_recipe_pdf(recipe, _make_ingredients(), servings=4)
        assert len(result) > 0

    def test_accented_ingredients(self) -> None:
        recipe = _make_recipe()
        ings = _make_ingredients([("crème fraîche", 200.0, "ml")])
        result = generate_recipe_pdf(recipe, ings, servings=4)
        assert len(result) > 0


# --- error handling ---------------------------------------------------------


class TestErrorHandling:
    def test_wraps_fpdf_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _broken_init(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(pdf, "FPDF", _broken_init)
        recipe = _make_recipe()

        with pytest.raises(PdfGenerationError, match="failed to generate PDF") as exc_info:
            generate_recipe_pdf(recipe, _make_ingredients(), servings=4)

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RuntimeError)


# --- logfire span -----------------------------------------------------------


@dataclass
class _FakeSpan:
    """Records span name, init kwargs, and set_attribute calls."""

    name: str
    init_kwargs: dict[str, Any]
    attributes: dict[str, Any] = field(default_factory=dict)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def __enter__(self) -> "_FakeSpan":
        return self

    def __exit__(self, *args: object) -> None:
        pass


class TestLogfireSpan:
    def test_span_attributes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        spans: list[_FakeSpan] = []

        @contextmanager
        def _fake_span(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
            s = _FakeSpan(name=name, init_kwargs=kwargs)
            spans.append(s)
            yield s

        fake_logfire = type("_FakeLogfire", (), {"span": staticmethod(_fake_span)})()
        monkeypatch.setattr(pdf, "logfire", fake_logfire)
        recipe = _make_recipe(recipe_id=42)
        generate_recipe_pdf(recipe, _make_ingredients(), servings=6)

        assert len(spans) == 1
        s = spans[0]
        assert s.name == "pdf.generate"
        assert s.init_kwargs["recipe_id"] == 42
        assert s.init_kwargs["servings"] == 6
        assert isinstance(s.attributes["duration_ms"], float)
        assert s.attributes["duration_ms"] >= 0
        assert isinstance(s.attributes["size_bytes"], int)
        assert s.attributes["size_bytes"] > 0


# --- performance SLA --------------------------------------------------------


class TestPerformance:
    def test_completes_under_five_seconds(self) -> None:
        """PDF generation must complete within 5 seconds."""
        steps = [f"Step {i}: Do something important." for i in range(1, 9)]
        recipe = _make_recipe(steps=steps)
        ings = _make_ingredients([(f"ingredient_{i}", float(i * 10), "g") for i in range(1, 11)])
        start = time.perf_counter()
        generate_recipe_pdf(recipe, ings, servings=8)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0
