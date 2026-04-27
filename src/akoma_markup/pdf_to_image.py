"""Render PDF pages to PIL images using pypdfium2.

Pure-Python wheels of Google's PDFium engine — same renderer Chrome ships, no
system binaries (poppler, ImageMagick) required. Replaces the pdf2image-based
path used by the multimodal POC in ``multimodal_analyzer.py``.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Iterable

from PIL import Image


def render_page(pdf_path: Path | str, page_num: int, dpi: int = 120) -> Image.Image:
    """Render a single 1-indexed PDF page to a PIL RGB image.

    Args:
        pdf_path: Path to the PDF.
        page_num: 1-indexed page number.
        dpi: Render DPI. 100-150 is plenty for vision-LLM classification;
            bump to 200+ if you need OCR-quality output downstream.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        if page_num < 1 or page_num > len(pdf):
            raise ValueError(
                f"page_num {page_num} out of range (PDF has {len(pdf)} pages)"
            )
        bitmap = pdf[page_num - 1].render(scale=dpi / 72)
        return bitmap.to_pil().convert("RGB")
    finally:
        pdf.close()


def render_pages(
    pdf_path: Path | str,
    page_nums: Iterable[int],
    dpi: int = 120,
) -> dict[int, Image.Image]:
    """Render multiple 1-indexed pages, returning ``{page_num: PIL.Image}``.

    Opens the PDF once and renders each requested page in order.
    """
    import pypdfium2 as pdfium

    page_nums = sorted(set(page_nums))
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        result: dict[int, Image.Image] = {}
        total = len(pdf)
        for p in page_nums:
            if p < 1 or p > total:
                raise ValueError(
                    f"page_num {p} out of range (PDF has {total} pages)"
                )
            bitmap = pdf[p - 1].render(scale=dpi / 72)
            result[p] = bitmap.to_pil().convert("RGB")
        return result
    finally:
        pdf.close()


def page_count(pdf_path: Path | str) -> int:
    """Return the number of pages in the PDF."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def image_to_data_url(img: Image.Image, format: str = "PNG") -> str:
    """Encode a PIL image as a base64 data URL for OpenAI-style vision APIs."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "png" if format.upper() == "PNG" else format.lower()
    return f"data:image/{mime};base64,{b64}"
