"""Optional table-rescue preprocessing for PDFs with embedded tables.

pdfplumber flattens tables into garbled text because column structure is lost.
This module re-extracts affected regions via Azure OCR (which emits tables as
clean markdown) and splices the result back into the per-page text list. The
downstream section-level LLM in converter.py then converts both prose AND
markdown tables to Laws.Africa markup in a single pass, so this module does no
LLM work itself for content — it only uses a vision LLM to *locate* table
pages in the ``auto`` mode.

Two strategies:
    - "declared" : user lists the table pages (cheapest, most reliable)
    - "auto"     : a vision LLM looks at every page image and flags tables;
                   only flagged pages get OCR'd
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Literal

from .pdf_to_image import render_pages
from .table_ocr_ai import AzureOCR
from .vision_llm import VisionClient

TableMode = Literal["declared", "auto", "full"]


_TABLE_DETECTION_PROMPT = (
    "Look at this single page from a legislative or regulatory document.\n"
    "Does it contain a TABLE — content arranged in clear rows and columns "
    "(ruled grids, schedule/form layouts, balance-sheet style two-column "
    "ledgers, or numerical tables with column headers)?\n"
    "Ignore: ordinary paragraphs, multi-column running prose, bulleted lists, "
    "section headings.\n"
    "Reply with a single token: YES or NO."
)


def parse_page_spec(spec: str) -> list[int]:
    """Parse a page spec like ``"10,12-15,20"`` into a sorted unique 1-indexed list.

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


def _detect_table_pages_vision(
    pdf_path: Path,
    total_pages: int,
    vision: VisionClient,
    dpi: int = 120,
    max_workers: int = 8,
) -> list[int]:
    """Render every page and ask the vision LLM which contain tables."""
    print(
        f"[tables] auto: rendering {total_pages} pages at {dpi} DPI ...",
        file=sys.stderr,
    )
    images = render_pages(pdf_path, range(1, total_pages + 1), dpi=dpi)
    print(
        f"[tables] auto: classifying {total_pages} pages with vision LLM "
        f"endpoint={vision.endpoint!r} deployment={vision.deployment!r} "
        f"(max_workers={max_workers}) ...",
        file=sys.stderr,
    )
    answers = vision.classify_pages(
        images, _TABLE_DETECTION_PROMPT, max_workers=max_workers
    )
    flagged: list[int] = []
    error_count = 0
    for p in sorted(answers):
        reply = answers[p]
        if reply.startswith("ERROR:"):
            error_count += 1
        is_table = reply.strip().upper().startswith("YES")
        print(
            f"[tables]   page {p}: {'TABLE' if is_table else 'no table'} "
            f"(reply={reply[:60]!r})",
            file=sys.stderr,
        )
        if is_table:
            flagged.append(p)
    if error_count == total_pages:
        raise RuntimeError(
            f"Vision LLM returned errors for ALL {total_pages} pages. "
            f"First error: {answers[sorted(answers)[0]]}. "
            f"Check AZURE_MULTIMODAL_ENDPOINT and AZURE_MULTIMODAL_DEPLOYMENT."
        )
    if error_count:
        print(
            f"[tables] auto: warning — {error_count}/{total_pages} pages "
            f"errored during vision classification",
            file=sys.stderr,
        )
    return flagged


def rescue_tables(
    pdf_path: Path,
    per_page_text: list[str],
    mode: TableMode,
    azure_api_key: str,
    table_pages: list[int] | None = None,
    azure_ocr_endpoint: str | None = None,
    azure_multimodal_endpoint: str | None = None,
    azure_multimodal_deployment: str | None = None,
    azure_multimodal_api_style: str | None = None,
) -> tuple[list[str], dict[int, str]]:
    """Produce a per-page text list where table-containing pages are re-extracted via OCR.

    Non-table pages stay as pdfplumber output; table pages are replaced with
    the Azure OCR markdown (which preserves tables as ``| … |`` pipe blocks).
    The downstream converter LLM turns those markdown tables into bluebell
    TABLE blocks as part of its normal per-section conversion pass.

    Args:
        pdf_path: Source PDF.
        per_page_text: pdfplumber output, one entry per page.
        mode: "declared" or "auto".
        azure_api_key: Azure API key for OCR (and vision in ``auto`` mode).
        table_pages: 1-indexed page list; required when mode == "declared".
        azure_ocr_endpoint: Override the Azure Document Intelligence endpoint.
        azure_multimodal_endpoint: Vision endpoint for ``auto`` mode. Falls
            back to ``AZURE_MULTIMODAL_ENDPOINT`` env var.
        azure_multimodal_deployment: Vision deployment for ``auto`` mode.
            Falls back to ``AZURE_MULTIMODAL_DEPLOYMENT`` env var.

    Returns:
        Tuple of (per_page_text, rescued_pages). per_page_text has OCR markdown
        spliced into rescued positions. rescued_pages maps 1-indexed page
        number -> OCR markdown for each rescued page, so the caller can
        guarantee that table content lands in the final output even if
        downstream section extraction fails to absorb it.
    """
    ocr = AzureOCR(api_key=azure_api_key, endpoint=azure_ocr_endpoint)

    if mode == "declared":
        if not table_pages:
            raise ValueError("mode='declared' requires table_pages=[...]")
        target_pages = table_pages
    elif mode == "auto":
        vision = VisionClient(
            api_key=azure_api_key,
            endpoint=azure_multimodal_endpoint,
            deployment=azure_multimodal_deployment,
            api_mode=azure_multimodal_api_style,
        )
        target_pages = _detect_table_pages_vision(
            pdf_path, len(per_page_text), vision
        )
        if not target_pages:
            print(
                "[tables] auto: vision LLM flagged no table pages, skipping OCR",
                file=sys.stderr,
            )
            return per_page_text, {}
        print(
            f"[tables] auto: vision LLM flagged pages {target_pages}",
            file=sys.stderr,
        )
    elif mode == "full":
        target_pages = list(range(1, len(per_page_text) + 1))
        print(
            f"[tables] full: OCR'ing all {len(target_pages)} pages "
            f"(no vision pre-flight)",
            file=sys.stderr,
        )
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
