"""Optional table-rescue preprocessing for PDFs with embedded tables.

pdfplumber flattens tables into garbled text because column structure is lost.
This module re-extracts affected regions via Azure OCR (which emits tables as
clean markdown), groups consecutive rescued pages into "regions" (one logical
table = one region, even when it spans pages), and splices a sentinel string
``<<TABLE_REGION:N>>`` into ``per_page_text`` at the position of each region.
The actual OCR markdown for each region is returned separately so callers
can:

  - keep table content OUT of the section-conversion LLM's input (the LLM
    sees only the small sentinel placeholder, not 200 rows of balance-sheet
    markdown that would blow its output budget), and
  - render each region deterministically via ``markdown_table.render_region``
    and splice the resulting bluebell ``TABLE`` block at the sentinel's
    position after the section LLM call returns.

Three detection strategies:
    - "declared" : user lists the table pages (cheapest, most reliable)
    - "auto"     : a vision LLM looks at every page image and flags tables;
                   only flagged pages get OCR'd
    - "full"     : OCR every page (no detection)
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Literal

from .pdf_to_image import render_pages
from .table_ocr_ai import AzureOCR
from .vision_llm import VisionClient

TableMode = Literal["declared", "auto", "full"]

# Sentinel format spliced into per_page_text. The section parser sees it
# as inert text; the section-conversion LLM is instructed to copy it
# verbatim; post-processing splices the rendered bluebell TABLE block in
# its place. The format must stay in sync with SENTINEL_RE below.
SENTINEL_FORMAT = "<<TABLE_REGION:{region_id}>>"
SENTINEL_RE = re.compile(r"<<TABLE_REGION:(\d+)>>")


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


def _cache_path(pdf_path: Path) -> Path:
    """Return the path of the per-page OCR cache file for ``pdf_path``."""
    cache_dir = pdf_path.parent / ".akoma_checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{pdf_path.stem}_table_ocr_cache.json"


def _load_ocr_cache(pdf_path: Path) -> dict[int, str]:
    """Load the per-page OCR cache, validated against the PDF's mtime.

    Returns ``{page_number: markdown}`` for pages already OCR'd in a prior
    run. Returns ``{}`` if the cache is missing, unreadable, or stale (the
    PDF has been modified since the cache was written).
    """
    cache_path = _cache_path(pdf_path)
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text())
    except Exception:  # noqa: BLE001
        return {}
    if data.get("pdf_mtime") != pdf_path.stat().st_mtime:
        print(
            f"[tables]   cache invalidated: PDF mtime changed",
            file=sys.stderr,
        )
        return {}
    return {int(k): v for k, v in data.get("pages", {}).items()}


def _save_ocr_cache(pdf_path: Path, page_md: dict[int, str]) -> None:
    """Atomically persist the per-page OCR cache to disk.

    Called after every successful page so a Ctrl+C during a long rescue run
    doesn't lose pages that already came back from Azure.
    """
    cache_path = _cache_path(pdf_path)
    payload = {
        "pdf_mtime": pdf_path.stat().st_mtime,
        "pdf_name": pdf_path.name,
        "pages": {str(k): v for k, v in sorted(page_md.items())},
    }
    tmp = cache_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(cache_path)


def _ocr_pages(pdf_path: Path, pages: list[int], ocr: AzureOCR) -> dict[int, str]:
    """OCR the given pages of a PDF and return {page_number: markdown}.

    One request per page so we get a clean page_number -> markdown mapping
    back and so we don't push a multi-megabyte merged PDF at the OCR
    endpoint (some Mistral document-ai deployments reject those with 400).

    Pages that were OCR'd in a prior run are loaded from
    ``.akoma_checkpoints/<stem>_table_ocr_cache.json`` and not re-fetched.
    The cache is saved incrementally after each successful page, so an
    interrupted run preserves all pages that completed before the
    interruption.
    """
    cache = _load_ocr_cache(pdf_path)
    cached_pages = [p for p in pages if p in cache]
    missing_pages = [p for p in pages if p not in cache]

    per_page: dict[int, str] = {p: cache[p] for p in cached_pages}
    if cached_pages:
        print(
            f"[tables]   OCR cache hit for {len(cached_pages)} page(s): "
            f"{cached_pages}",
            file=sys.stderr,
        )
    if not missing_pages:
        return per_page
    print(
        f"[tables]   OCR cache miss for {len(missing_pages)} page(s): "
        f"{missing_pages}",
        file=sys.stderr,
    )

    for i, p in enumerate(missing_pages, start=1):
        single = _slice_pdf(pdf_path, [p])
        try:
            print(
                f"[tables]   OCR page {p} ({i}/{len(missing_pages)}) ...",
                file=sys.stderr,
            )
            md = ocr.extract_text(single)
            per_page[p] = md
            cache[p] = md
            _save_ocr_cache(pdf_path, cache)  # checkpoint after each page
            print(
                f"[tables] --- OCR markdown for page {p} ---\n"
                f"{md}\n"
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


def _group_consecutive_pages(pages: list[int]) -> list[list[int]]:
    """Collapse a sorted page list into consecutive runs.

    ``[3, 78, 79, 80, 92]`` -> ``[[3], [78, 79, 80], [92]]``. A multi-page
    table normally occupies consecutive pages, so consecutive rescued pages
    = one logical region.
    """
    if not pages:
        return []
    pages = sorted(set(pages))
    groups: list[list[int]] = [[pages[0]]]
    for p in pages[1:]:
        if p == groups[-1][-1] + 1:
            groups[-1].append(p)
        else:
            groups.append([p])
    return groups


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
) -> tuple[list[str], list[dict]]:
    """Detect, OCR, and isolate tabular regions from a PDF's per-page text.

    Returns ``per_page_text`` with each rescued region replaced by a
    ``<<TABLE_REGION:N>>`` sentinel (so the section parser doesn't drag
    table markdown into a section's content blob and blow the LLM's output
    budget), plus a list of region dicts containing the actual OCR markdown.

    Args:
        pdf_path: Source PDF.
        per_page_text: pdfplumber output, one entry per page.
        mode: "declared", "auto", or "full".
        azure_api_key: Azure API key for OCR (and vision in ``auto`` mode).
        table_pages: 1-indexed page list; required when mode == "declared".
        azure_ocr_endpoint: Override the Azure Document Intelligence endpoint.
        azure_multimodal_endpoint: Vision endpoint for ``auto`` mode.
        azure_multimodal_deployment: Vision deployment for ``auto`` mode.
        azure_multimodal_api_style: Vision API style for ``auto`` mode
            ('chat' | 'responses' | 'azure-inference').

    Returns:
        Tuple ``(per_page_text, table_regions)`` where:
          - ``per_page_text`` has the first page of each region replaced by
            the sentinel string and the rest of the region's pages blanked,
          - ``table_regions`` is a list of dicts ``{id, pages, markdown}``,
            one per region, in document order. Each region's ``markdown`` is
            the concatenated OCR output for its constituent pages.
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
            return per_page_text, []
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

    page_groups = _group_consecutive_pages(target_pages)
    print(
        f"[tables] grouped {len(target_pages)} page(s) into "
        f"{len(page_groups)} region(s): {page_groups}",
        file=sys.stderr,
    )

    result = list(per_page_text)
    table_regions: list[dict] = []
    for region_id, pages in enumerate(page_groups):
        region_md = "\n\n".join(
            ocr_markdown[p] for p in pages if p in ocr_markdown
        ).strip()
        if not region_md:
            continue
        sentinel = SENTINEL_FORMAT.format(region_id=region_id)
        # Sentinel goes on the first page of the region (so it occupies that
        # page's slot in the document order). Other pages in the region are
        # blanked so they don't contribute pdfplumber junk to raw_text.
        result[pages[0] - 1] = f"\n{sentinel}\n"
        for p in pages[1:]:
            result[p - 1] = ""
        table_regions.append(
            {"id": region_id, "pages": pages, "markdown": region_md}
        )

    return result, table_regions
