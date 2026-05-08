"""akoma-markup: Convert legislative PDFs to Akoma Ntoso markup."""

import re
import sys
from pathlib import Path

from . import debug_dump
from .conversion import build_chain, process_all_sections
from .util.pdf.text import extract_pdf_pages
from .util.llm.factory import build_llm
from .parsing.tables.render import render_region
from .parsing.text.chapter_section_mapping import (
    extract_chapter_ranges,
    extract_section_content,
    filter_sections_by_chapters,
    parse_toc,
)
from .output import write_markup, write_metadata


# Recognises the rendered ``<<TABLE_REGION:N>>`` sentinel on its own line (with arbitrary indentation).
_SENTINEL_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)<<TABLE_REGION:(?P<id>\d+)>>[ \t]*$",
    re.MULTILINE,
)


def _splice_sentinels(
    markup: str, table_blocks: dict[int, str]
) -> tuple[str, set[int]]:
    """Replace each ``<<TABLE_REGION:N>>`` sentinel line with its TABLE block.
    while maintaining the original indentation for the TABLE block. Returns the 
    rewritten markup and the set of region IDs that were consumed (so the caller 
    can identify trailing regions that need to be emitted as standalone TABLE blocks).
    """
    consumed: set[int] = set()

    def _replace(m: re.Match) -> str:
        indent = m.group("indent")
        region_id = int(m.group("id"))
        if region_id not in table_blocks:
            return m.group(0)  # unknown id; leave token in place
        consumed.add(region_id)
        block = table_blocks[region_id]
        return "\n".join(indent + ln if ln else "" for ln in block.split("\n"))

    return _SENTINEL_LINE_RE.sub(_replace, markup), consumed


def _extract_and_rescue_tables(
    pdf: Path,
    table_mode: str | None,
    table_pages: list[int] | None,
    azure_vision_key: str | None,
    azure_vision_endpoint: str | None,
    azure_vision_model: str | None,
    azure_vision_api_style: str | None,
    azure_vision_max_tokens: int | None,
) -> tuple[list[str], list[dict], dict[int, str]]:
    """Extract PDF text and optionally rescue tables via vision LLM."""
    print("Extracting text from PDF ...", file=sys.stderr)
    per_page_text = extract_pdf_pages(str(pdf))

    table_regions: list[dict] = []
    table_blocks: dict[int, str] = {}
    if table_mode is not None:
        from .parsing.tables.rescue import rescue_tables
        print(f"Rescuing tables via vision LLM (mode={table_mode!r}) ...", file=sys.stderr)
        per_page_text, table_regions = rescue_tables(
            pdf_path=pdf,
            per_page_text=per_page_text,
            mode=table_mode,
            azure_vision_key=azure_vision_key,
            azure_vision_endpoint=azure_vision_endpoint,
            azure_vision_model=azure_vision_model,
            azure_vision_api_style=azure_vision_api_style,
            table_pages=table_pages,
            azure_vision_max_tokens=azure_vision_max_tokens,
        )
        table_blocks = {r["id"]: render_region(r["markdown"]) for r in table_regions}
        print(f"[tables] rescued {len(table_regions)} region(s); rendered {len(table_blocks)} bluebell TABLE blocks", file=sys.stderr)

    return per_page_text, table_regions, table_blocks


def _parse_document_structure(
    per_page_text: list[str],
    output_path: str,
    pdf: Path,
) -> list[dict]:
    """Parse TOC, extract chapters and sections from raw text."""
    raw_text = "\n".join(per_page_text)
    debug_dump.write_raw_text(raw_text, output_path)

    all_lines = raw_text.splitlines()
    print("Parsing table of contents ...", file=sys.stderr)
    _chapters, section_names, toc_end_line = parse_toc(all_lines)
    chapter_ranges = extract_chapter_ranges(all_lines, section_names, toc_end_line)
    print(f"Found {len(chapter_ranges)} chapters, {len(section_names)} sections (TOC ends at line {toc_end_line})", file=sys.stderr)

    content_text = "\n".join(all_lines[toc_end_line + 1:])
    sections = extract_section_content(content_text, section_names)
    sections = filter_sections_by_chapters(sections, chapter_ranges)

    # Remove duplicate sections
    seen = set()
    unique = []
    for sec in sections:
        if sec["num"] not in seen:
            seen.add(sec["num"])
            unique.append(sec)
    sections = unique
    print(f"{len(sections)} unique sections ready for conversion", file=sys.stderr)

    debug_dump.write_parser_summary(pdf, toc_end_line, section_names, chapter_ranges, output_path)
    debug_dump.write_sections_tsv(sections, output_path)
    debug_dump.write_ocr_text(content_text, output_path)

    return sections


def _process_conversion(
    sections: list[dict],
    llm,
    document_name: str,
    output_path: str,
    pdf: Path,
    table_regions: list[dict],
    table_blocks: dict[int, str],
) -> tuple[list[dict], int]:
    """Convert sections via LLM and splice table regions."""
    chain = build_chain(llm, document_name=document_name)
    checkpoint_dir = Path(output_path).parent / ".akoma_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_filename = f"{pdf.stem}_conversion_checkpoint.json"
    checkpoint_path = checkpoint_dir / checkpoint_filename
    converted, errors = process_all_sections(chain, sections, checkpoint_path=checkpoint_path)

    # Add chapter info to converted sections
    sec_lookup = {s["num"]: s for s in sections}
    for conv in converted:
        orig = sec_lookup.get(conv["num"], {})
        conv["chapter_roman"] = orig.get("chapter_roman", "NA")
        conv["chapter_heading"] = orig.get("chapter_heading", "Unknown")
        conv["kind"] = "section"

    # Splice table sentinels with bluebell TABLE blocks
    consumed_region_ids: set[int] = set()
    if table_blocks:
        for conv in converted:
            spliced, consumed = _splice_sentinels(conv["markup"], table_blocks)
            conv["markup"] = spliced
            consumed_region_ids |= consumed
        print(f"[tables] spliced {len(consumed_region_ids)}/{len(table_blocks)} region(s) into section markup", file=sys.stderr)

    # Add trailing table regions as standalone blocks
    trailing_regions = [r for r in table_regions if r["id"] not in consumed_region_ids]
    if trailing_regions:
        print(f"[tables] {len(trailing_regions)} region(s) had no enclosing section; emitting as top-level TABLE block(s)", file=sys.stderr)
        for r in trailing_regions:
            block = table_blocks.get(r["id"], "")
            if not block:
                continue
            converted.append({
                "num": f"TBL_R{r['id']}",
                "markup": block,
                "kind": "trailing_table",
                "pages": r["pages"],
            })

    return converted, errors


def convert(
    pdf_path: str,
    llm_config: dict,
    output_path: str | None = None,
    document_name: str | None = None,
    act_number: str | None = None,
    replaces: str | None = None,
    table_mode: str | None = None,
    table_pages: list[int] | None = None,
    azure_vision_key: str | None = None,
    azure_vision_endpoint: str | None = None,
    azure_vision_model: str | None = None,
    azure_vision_api_style: str | None = None,
    azure_vision_max_tokens: int | None = None,
) -> str:
    """Convert a legislative PDF to Akoma Ntoso markup.

    Args:
        pdf_path: Path to the legislative PDF file.
        llm_config: LLM provider config dict. Must include 'provider' key.
            Example: {"provider": "openai", "model": "gpt-4o", "api_key": "sk-..."}
        output_path: Destination for the markup file.
            Defaults to ``<pdf_stem>_markup.txt`` in the same directory.
        document_name: Name of the document (e.g., "Bharatiya Nagarik Suraksha Sanhita 2023").
            Defaults to PDF filename stem.
        act_number: Act number (e.g., "46 of 2023").
        replaces: Previous act this document replaces (e.g., "Criminal Procedure Code (CrPC) 1973").
        table_mode: Optional table-rescue strategy. One of "declared", "auto",
            or "full". When None (default), only pdfplumber is used and tables
            in the PDF may be garbled in the output. When set, the selected
            pages are re-extracted via the vision LLM (which renders the page
            as markdown with pipe-format tables) and converted to Laws.Africa
            TABLE blocks. Requires the four ``azure_vision_*`` arguments
            below.
        table_pages: 1-indexed page list. Required when `table_mode="declared"`.
        azure_vision_key: Vision-LLM API key. Required when `table_mode` is set.
        azure_vision_endpoint: Vision-LLM endpoint. Required when `table_mode` is set.
        azure_vision_model: Vision-LLM model/deployment name. Required when
            `table_mode` is set.
        azure_vision_api_style: Vision-LLM API style — one of 'chat',
            'responses', 'azure-inference'. Required when `table_mode` is set.
        azure_vision_max_tokens: Per-page output token budget for the
            extraction call. Defaults to ``AZURE_VISION_MAX_TOKENS`` env var
            or 16384. Bump this if you see truncation warnings on dense
            schedule pages.

    Returns:
        Path to the generated markup file.
    """
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    if output_path is None:
        output_path = str(pdf.with_name(f"{pdf.stem}_markup.txt"))

    if document_name is None:
        document_name = pdf.stem

    if table_mode is not None:
        if table_mode not in {"declared", "auto", "full"}:
            raise ValueError(f"table_mode must be 'declared', 'auto', or 'full'; got {table_mode!r}")
        if not azure_vision_key:
            raise ValueError("table_mode requires azure_vision_key")
        if not azure_vision_endpoint:
            raise ValueError("table_mode requires azure_vision_endpoint")
        if not azure_vision_model:
            raise ValueError("table_mode requires azure_vision_model")
        if not azure_vision_api_style:
            raise ValueError("table_mode requires azure_vision_api_style")
        if table_mode == "declared" and not table_pages:
            raise ValueError("table_mode='declared' requires table_pages")

    llm = build_llm(llm_config)

    # Step 1: Extract text and rescue tables if needed
    per_page_text, table_regions, table_blocks = _extract_and_rescue_tables(
        pdf, table_mode, table_pages, azure_vision_key, azure_vision_endpoint,
        azure_vision_model, azure_vision_api_style, azure_vision_max_tokens
    )

    # Step 2: Parse document structure
    sections = _parse_document_structure(per_page_text, output_path, pdf)

    # Step 3: Process conversion via LLM
    converted, errors = _process_conversion(
        sections, llm, document_name, output_path, pdf, table_regions, table_blocks
    )

    # Step 4: Write final outputs
    markup_path = write_markup(converted, output_path)
    meta_path = write_metadata(
        converted, errors, output_path,
        document_name=document_name,
        act_number=act_number,
        replaces=replaces
    )

    print(f"Markup written to {markup_path}", file=sys.stderr)
    print(f"Metadata written to {meta_path}", file=sys.stderr)
    if errors:
        print(f"{len(errors)} sections failed conversion", file=sys.stderr)

    return markup_path