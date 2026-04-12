"""fpdf2 PDF generation service for recipe cards.

Exposes :func:`generate_recipe_pdf`, a synchronous function that turns a
:class:`models.Recipe` with its :class:`models.Ingredient` list and a target
servings count into a formatted A4 PDF returned as ``bytes``.

The function is CPU-bound but fast — typical recipes render in under 50 ms —
so no ``asyncio.to_thread`` wrapper is applied here. The caller in ``bot.py``
can offload to a thread if desired.

Layout
------
A4 portrait (210 x 297 mm), 15 mm margins.  DejaVu Sans TTF throughout (full
Unicode support for French text including ``oe`` ligature, em-dash, etc.).
Sections: title (bold 18 pt, centred), servings subtitle (italic 12 pt,
centred), ingredients with quantity adjustment, and numbered preparation steps
with ``multi_cell`` wrapping.

Quantity adjustment
-------------------
``new_qty = original_qty * servings / recipe.servings``, formatted via
:func:`models.format_quantity` (``:.2f`` with trailing-zero stripping).
``None`` quantities and units are handled gracefully.
"""

import json
import time
from pathlib import Path

import logfire
from fpdf import FPDF

from models import Ingredient, Recipe, format_ingredient_line

_DEJAVU_DIR = Path("/usr/share/fonts/truetype/dejavu")
_HAS_DEJAVU = (_DEJAVU_DIR / "DejaVuSans.ttf").is_file()
_FONT_FAMILY = "DejaVuSans" if _HAS_DEJAVU else "Helvetica"

__all__ = ["PdfError", "PdfGenerationError", "generate_recipe_pdf"]


class PdfError(Exception):
    """Base class for any failure inside the PDF generation service.

    The global error handler catches this and maps it to the French
    user message ``"❌ Erreur lors de la génération du PDF."``.
    """


class PdfGenerationError(PdfError):
    """fpdf2 raised during PDF rendering."""


def generate_recipe_pdf(
    recipe: Recipe,
    ingredients: list[Ingredient],
    servings: int,
) -> bytes:
    """Render *recipe* as a formatted A4 PDF with quantities adjusted for *servings*.

    Parameters
    ----------
    recipe:
        The persisted recipe.  Only ``.id``, ``.title``, ``.servings``, and
        ``.steps`` (JSON string) are read.
    ingredients:
        Ingredient rows for this recipe — passed explicitly rather than via
        ``recipe.ingredients`` so the caller has full control over ordering
        and filtering.
    servings:
        Target number of servings.  Ingredient quantities are scaled by
        ``servings / recipe.servings``.

    Returns
    -------
    bytes
        The raw PDF file content (starts with ``%PDF``).

    Raises
    ------
    PdfGenerationError
        Any fpdf2 or JSON error during rendering.
    """
    with logfire.span("pdf.generate", recipe_id=recipe.id, servings=servings) as span:
        start = time.perf_counter()
        try:
            pdf = FPDF()
            if _HAS_DEJAVU:
                pdf.add_font(_FONT_FAMILY, style="", fname=str(_DEJAVU_DIR / "DejaVuSans.ttf"))
                pdf.add_font(
                    _FONT_FAMILY, style="B", fname=str(_DEJAVU_DIR / "DejaVuSans-Bold.ttf")
                )
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)

            # Title
            pdf.set_font(_FONT_FAMILY, "B", 18)
            pdf.cell(0, 10, recipe.title, new_x="LMARGIN", new_y="NEXT", align="C")

            # Servings
            pdf.set_font(_FONT_FAMILY, "", 12)
            pdf.cell(
                0,
                8,
                f"Pour {servings} personnes",
                new_x="LMARGIN",
                new_y="NEXT",
                align="C",
            )
            pdf.ln(5)

            # Ingredients
            pdf.set_font(_FONT_FAMILY, "B", 14)
            pdf.cell(0, 10, "Ingrédients", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font(_FONT_FAMILY, "", 11)
            for ing in ingredients:
                line = format_ingredient_line(
                    ing.name, ing.quantity, ing.unit, servings, recipe.servings
                )
                pdf.cell(
                    0,
                    7,
                    f"  - {line}",
                    new_x="LMARGIN",
                    new_y="NEXT",
                )
            pdf.ln(5)

            # Steps
            pdf.set_font(_FONT_FAMILY, "B", 14)
            pdf.cell(0, 10, "Préparation", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font(_FONT_FAMILY, "", 11)
            steps: list[str] = json.loads(recipe.steps)
            for i, step in enumerate(steps, 1):
                pdf.multi_cell(0, 7, f"{i}. {step}")
                pdf.ln(2)

            raw = pdf.output()
        except Exception as exc:
            raise PdfGenerationError("failed to generate PDF") from exc

        if raw is None:
            raise PdfGenerationError("fpdf2 returned None from output()")
        result = bytes(raw)

        duration_ms = (time.perf_counter() - start) * 1000
        span.set_attribute("duration_ms", duration_ms)
        span.set_attribute("size_bytes", len(result))

    return result
