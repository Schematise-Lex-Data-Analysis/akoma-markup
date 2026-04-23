"""Optional table-rescue preprocessing for PDFs with embedded tables.

pdfplumber flattens tables into garbled text because column structure is lost.
This module re-extracts affected regions via Azure OCR (which emits tables as
clean markdown) and splices the result back into the per-page text list. The
downstream section-level LLM in converter.py then converts both prose AND
markdown tables to Laws.Africa markup in a single pass, so this module does no
LLM work itself — it is purely an extractor.

Three strategies:
    - "declared"  : user lists the table pages (cheapest, most reliable)
    - "heuristic" : pdfplumber.find_tables() flags ruled-table pages, OCR those
    - "full"      : OCR every page, use OCR markdown as the whole text source
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Literal

from .table_ocr_ai import AzureOCR

TableMode = Literal["declared", "heuristic", "full"]


def parse_page_spec(spec: str) -> list[int]:
    """Parse a page spec like "10,12-15,20" into a sorted unique list of 1-indexed page numbers.

    Args:
        spec: Comma-separated page numbers and ranges.

    Returns:
        Sorted list of unique 1-indexed page numbers.

    Raises:
        ValueError: If the spec is malformed.
    """
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo < 1 or hi < lo:
                raise ValueError(f"Invalid page range: {part!r}")
            pages.update(range(lo, hi + 1))
        else:
            n = int(part)
            if n < 1:
                raise ValueError(f"Page numbers must be >= 1: {part!r}")
            pages.add(n)
    return sorted(pages)


def _slice_pdf(pdf_path: Path, pages: list[int]) -> Path:
    """Write a temp PDF containing only the given 1-indexed pages. Caller must delete."""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    for p in pages:
        if p < 1 or p > len(reader.pages):
            raise ValueError(
                f"Requested page {p} but PDF has {len(reader.pages)} pages"
            )
        writer.add_page(reader.pages[p - 1])

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    with tmp as f:
        writer.write(f)
    return Path(tmp.name)


def _ocr_pages(pdf_path: Path, pages: list[int], ocr: AzureOCR) -> dict[int, str]:
    """OCR the given pages of a PDF and return {page_number: markdown}.

    One request per page so we get a clean page_number -> markdown mapping back
    and so we don't push a multi-megabyte merged PDF at the OCR endpoint (which
    some Mistral document-ai deployments reject with 400).
    """
    per_page: dict[int, str] = {}
    for i, p in enumerate(pages, start=1):
        single = _slice_pdf(pdf_path, [p])
        try:
            print(
                f"[tables]   OCR page {p} ({i}/{len(pages)}) ...",
                file=sys.stderr,
            )
            per_page[p] = ocr.extract_text(single)
            print(
                f"[tables] --- OCR markdown for page {p} ---\n"
                f"{per_page[p]}\n"
                f"[tables] --- end page {p} ---",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001 — record and continue
            print(
                f"[tables]   OCR failed for page {p}: {exc}",
                file=sys.stderr,
            )
            raise
        finally:
            single.unlink(missing_ok=True)
    return per_page


def _classify_table_pages(pdf_path: Path) -> list[int]:
    """Return 1-indexed page numbers that contain at least one table.

    Uses pdfplumber.find_tables() with the default (line-based) strategy —
    reliable and false-positive-free on PDFs with ruled tables. PDFs that
    rely on whitespace alignment (no visible rules) will flag few/no pages;
    use `--table-mode declared` for those.
    """
    import pdfplumber

    flagged: list[int] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            if page.find_tables():
                flagged.append(idx)
    return flagged


def rescue_tables(
    pdf_path: Path,
    per_page_text: list[str],
    mode: TableMode,
    azure_api_key: str,
    table_pages: list[int] | None = None,
    azure_ocr_endpoint: str | None = None,
) -> tuple[list[str], dict[int, str]]:
    """Produce a per-page text list where table-containing pages are re-extracted via OCR.

    Non-table pages stay as pdfplumber output; table pages are replaced with the
    Azure OCR markdown (which preserves tables as `| … |` pipe blocks). The
    downstream converter LLM turns those markdown tables into bluebell TABLE
    blocks as part of its normal per-section conversion pass.

    Args:
        pdf_path: Source PDF.
        per_page_text: pdfplumber output, one entry per page.
        mode: "declared" | "heuristic" | "full".
        azure_api_key: Azure API key for OCR.
        table_pages: 1-indexed page list; required when mode == "declared".
        azure_ocr_endpoint: Override the Azure Document Intelligence endpoint.

    Returns:
        Tuple of (per_page_text, rescued_pages). per_page_text has OCR markdown
        spliced into rescued positions. rescued_pages maps 1-indexed page number
        -> OCR markdown for each rescued page, so the caller can guarantee that
        table content lands in the final output even if downstream section
        extraction fails to absorb it.
    """
    ocr = AzureOCR(api_key=azure_api_key, endpoint=azure_ocr_endpoint)

    if mode == "declared":
        if not table_pages:
            raise ValueError("mode='declared' requires table_pages=[...]")
        target_pages = table_pages
    elif mode == "heuristic":
        target_pages = _classify_table_pages(pdf_path)
        if not target_pages:
            print(
                "[tables] heuristic: no tables detected by pdfplumber, skipping OCR",
                file=sys.stderr,
            )
            return per_page_text, {}
        print(
            f"[tables] heuristic: found tables on pages {target_pages}",
            file=sys.stderr,
        )
    elif mode == "full":
        target_pages = list(range(1, len(per_page_text) + 1))
    else:
        raise ValueError(f"Unknown table mode: {mode!r}")

    print(
        f"[tables] OCR'ing {len(target_pages)} page(s) in mode={mode!r}",
        file=sys.stderr,
    )
    ocr_markdown = _ocr_pages(pdf_path, target_pages, ocr)

    result = list(per_page_text)
    for page_num, md in ocr_markdown.items():
        result[page_num - 1] = md

    return result, ocr_markdown
