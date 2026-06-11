"""akoma-markup: Convert legislative PDFs to Akoma Ntoso markup."""

import json
import logging
import re
from datetime import datetime
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
from .amendment.conversion import (
    build_gazette_chain,
    convert_gazette_text,
    convert_gazette_pages_with_vision,
    merge_gazette_pages,
)

logger = logging.getLogger(__name__)


def _log_step(title: str) -> None:
    """Emit a visually distinct step heading on a new line."""
    logger.info("\n══ %s ══", title)


# Recognises the rendered ``<<TABLE_REGION:N>>`` sentinel on its own line (with arbitrary indentation).
_SENTINEL_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)<<TABLE_REGION:(?P<id>\d+)>>[ \t]*$",
    re.MULTILINE,
)


def _splice_sentinels(
    markup: str, table_blocks: dict[int, str]
) -> tuple[str, set[int]]:
    """Replace each ``<<TABLE_REGION:N>>`` sentinel line with its TABLE block,
    while maintaining the original indentation for the TABLE block. 
    Returns the rewritten markup and the set of region IDs that were consumed (so the caller 
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
    output_path: str,
    table_mode: str | None,
    table_pages: list[int] | None,
    azure_vision_key: str | None,
    azure_vision_endpoint: str | None,
    azure_vision_model: str | None,
    azure_vision_api_style: str | None,
    azure_vision_max_tokens: int | None,
) -> tuple[list[str], list[dict], dict[int, str]]:
    """Extract PDF text and optionally rescue tables via vision LLM."""
    _log_step("Extracting text from PDF")
    per_page_text = extract_pdf_pages(str(pdf))

    table_regions: list[dict] = []
    table_blocks: dict[int, str] = {}
    if table_mode is not None:
        from .parsing.tables.rescue import rescue_tables
        _log_step(f"Rescuing tables via vision LLM (mode={table_mode!r})")
        per_page_text, table_regions = rescue_tables(
            pdf_path=pdf,
            per_page_text=per_page_text,
            output_path=output_path,
            mode=table_mode,
            azure_vision_key=azure_vision_key,
            azure_vision_endpoint=azure_vision_endpoint,
            azure_vision_model=azure_vision_model,
            azure_vision_api_style=azure_vision_api_style,
            table_pages=table_pages,
            azure_vision_max_tokens=azure_vision_max_tokens,
        )
        table_blocks = {r["id"]: render_region(r["markdown"]) for r in table_regions}
        logger.info(
            "Rescued %d region(s); rendered %d bluebell TABLE blocks",
            len(table_regions), len(table_blocks),
        )
        if table_regions:
            debug_dump.write_table_regions(
                pdf, table_regions, table_blocks, output_path
            )

    return per_page_text, table_regions, table_blocks


def _parse_document_structure(
    per_page_text: list[str],
    output_path: str,
    pdf: Path,
) -> list[dict]:
    """Parse TOC, extract chapters and sections from raw text."""
    _log_step("Parsing table of contents")
    raw_text = "\n".join(per_page_text)
    debug_dump.write_raw_text(pdf, raw_text, output_path)

    all_lines = raw_text.splitlines()
    _chapters, section_names, toc_end_line = parse_toc(all_lines)
    chapter_ranges = extract_chapter_ranges(all_lines, section_names, toc_end_line)
    logger.info(
        "Found %d chapters, %d sections (TOC ends at line %d)",
        len(chapter_ranges), len(section_names), toc_end_line,
    )

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
    logger.info("%d unique sections ready for conversion", len(sections))

    debug_dump.write_parser_summary(pdf, toc_end_line, section_names, chapter_ranges, output_path)
    debug_dump.write_sections_tsv(pdf, sections, output_path)
    debug_dump.write_ocr_text(pdf, content_text, output_path)

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
    _log_step("Converting sections to Akoma Ntoso")
    chain = build_chain(llm, document_name=document_name)
    checkpoint_dir = Path(output_path).parent / ".akoma_cache"
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
        logger.info(
            "Spliced %d/%d table region(s) into section markup",
            len(consumed_region_ids), len(table_blocks),
        )

    # Add trailing table regions as standalone blocks
    trailing_regions = [r for r in table_regions if r["id"] not in consumed_region_ids]
    if trailing_regions:
        logger.info(
            "%d table region(s) had no enclosing section; emitting as top-level block(s)",
            len(trailing_regions),
        )
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
        pdf, output_path, table_mode, table_pages,
        azure_vision_key, azure_vision_endpoint, azure_vision_model,
        azure_vision_api_style, azure_vision_max_tokens,
    )

    # Step 2: Parse document structure
    sections = _parse_document_structure(per_page_text, output_path, pdf)

    # Step 3: Process conversion via LLM Inferencing
    converted, errors = _process_conversion(
        sections, llm, document_name, output_path, pdf, table_regions, table_blocks
    )

    # Step 4: Write final outputs
    _log_step("Writing final outputs")
    markup_path = write_markup(converted, output_path)
    meta_path = write_metadata(
        converted, errors, output_path,
        document_name=document_name,
        act_number=act_number,
        replaces=replaces
    )

    logger.info("Markup written to %s", markup_path)
    logger.info("Metadata written to %s", meta_path)
    if errors:
        logger.warning("%d sections failed conversion", len(errors))

    return markup_path


def convert_gazette(
    gazette_pdf: str | Path,
    output_path: str,
    *,
    llm_config: dict,
    document_name: str = "Gazette Notification",
    use_vision: bool = True,
    azure_vision_key: str | None = None,
    azure_vision_endpoint: str | None = None,
    azure_vision_model: str | None = None,
    azure_vision_api_style: str | None = "chat",
    azure_vision_max_tokens: int | None = None,
) -> str:
    """Convert a gazette PDF to markup.

    Args:
        gazette_pdf: Path to gazette PDF file
        output_path: Output markup file path
        llm_config: LLM configuration dictionary. Must include 'provider' key.
            Example: {"provider": "azure", "model": "gpt-4o"}
        document_name: Name for the gazette document
        use_vision: If True, use multimodal vision LLM to process pages.
            Otherwise, extract text and process with text-based LLM.
        azure_vision_key: Azure OpenAI API key for vision model (if use_vision)
        azure_vision_endpoint: Azure OpenAI endpoint (if use_vision)
        azure_vision_model: Azure OpenAI model/deployment (if use_vision)
        azure_vision_api_style: API style - 'chat', 'responses', or 'azure-inference'
        azure_vision_max_tokens: Max tokens for vision model responses

    Returns:
        Path to the generated markup file.
    """
    pdf = Path(gazette_pdf)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    # Build LLM from config
    llm = build_llm(llm_config)

    if use_vision:
        _log_step("Converting gazette using multimodal vision LLM")

        # Initialize vision client
        from .util.llm.vision import VisionClient

        vision_client = VisionClient(
            api_key=azure_vision_key,
            endpoint=azure_vision_endpoint,
            deployment=azure_vision_model,
            api_mode=azure_vision_api_style,
            extraction_max_tokens=azure_vision_max_tokens,
        )

        # Render all pages to images
        _log_step("Rendering PDF pages to images")
        from .util.pdf.images import render_pages, page_count

        total_pages = page_count(pdf)
        page_nums = list(range(1, total_pages + 1))
        # Increase DPI for better text extraction - vision models need clear text
        page_images = render_pages(pdf, page_nums, dpi=300)

        logger.info("Rendered %d pages", len(page_images))

        # Process pages with vision
        checkpoint_dir = Path(output_path).parent / ".akoma_cache"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{pdf.stem}_gazette_checkpoint.json"

        page_markups = convert_gazette_pages_with_vision(
            vision_client=vision_client,
            page_images=page_images,
            checkpoint_path=checkpoint_path,
        )

        # Merge page markups
        _log_step("Merging page markups")
        merged_markup = merge_gazette_pages(page_markups, llm=None)

    else:
        _log_step("Converting gazette using text extraction")

        # Extract text from PDF
        from .util.pdf.text import extract_pdf_pages

        per_page_text = extract_pdf_pages(str(pdf))
        gazette_text = "\n".join(per_page_text)

        # Build chain and convert
        chain = build_gazette_chain(llm, document_name=document_name)
        merged_markup = convert_gazette_text(chain, gazette_text)

    # Write output
    _log_step(f"Writing output to {output_path}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(merged_markup)

    # Write metadata
    metadata = {
        "document_name": document_name,
        "source_file": str(gazette_pdf),
        "conversion_date": datetime.now().isoformat(),
        "type": "gazette",
        "use_vision": use_vision,
    }
    meta_path = Path(output_path).with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Gazette conversion complete")
    logger.info("Markup written to %s", output_path)
    logger.info("Metadata written to %s", meta_path)

    return str(output_path)


def convert_gazette_from_tsv(
    tsv_path: str | Path,
    output_path: str | Path,
    *,
    llm_config: dict,
    document_name: str | None = None,
) -> str:
    """Convert a Gazette TSV (from gazette-extract) to markup.

    This is Phase 3 of the Gazette pipeline: Takes the TSV output from
    gazette-extract and uses LLM to convert each section to Akoma Ntoso
    markup.

    Args:
        tsv_path: Path to the TSV file from gazette-extract
        output_path: Output markup file path
        llm_config: LLM configuration dictionary. Must include 'provider' key.
            Example: {"provider": "azure", "model": "gpt-4o"}
        document_name: Name for the document (defaults to TSV stem)

    Returns:
        Path to the generated markup file.
    """
    import pandas as pd
    from pathlib import Path
    from .gazette.conversion import (
        build_gazette_chain,
        process_gazette_sections,
        assemble_gazette_markup,
    )

    tsv = Path(tsv_path)
    if not tsv.exists():
        raise FileNotFoundError(f"TSV not found: {tsv}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load TSV
    _log_step(f"Loading TSV: {tsv}")
    tsv_df = pd.read_csv(tsv, sep="\t")
    logger.info("Loaded %d sections", len(tsv_df))

    # Determine document name
    doc_name = document_name or tsv.stem.replace("_sections", "")

    # Build LLM and chain
    _log_step("Building LLM chain")
    llm = build_llm(llm_config)
    chain = build_gazette_chain(llm, doc_name)

    # Prepare sections from DataFrame
    sections = []
    for _, row in tsv_df.iterrows():
        sections.append({
            "num": str(row["section_num"]),
            "heading": str(row["heading"]) if pd.notna(row["heading"]) else "",
            "content": str(row["content"]) if pd.notna(row["content"]) else "",
            "hierarchy_level": int(row.get("hierarchy_level", 1))
            if pd.notna(row.get("hierarchy_level")) else 1,
            "parent_section": str(row["parent_section"])
            if pd.notna(row.get("parent_section")) else "",
            "page": int(row["page"]) if pd.notna(row["page"]) else 0,
        })

    # Setup checkpoint path
    checkpoint_dir = output.parent / ".akoma_cache"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{output.stem}_markup_checkpoint.json"

    # Process all sections with LLM
    _log_step("Converting sections to markup")
    results, errors = process_gazette_sections(
        chain=chain,
        sections=sections,
        checkpoint_path=checkpoint_path,
    )

    # Log errors
    if errors:
        logger.warning("%d sections failed conversion", len(errors))
        for err in errors:
            logger.warning("  Section %s: %s", err["num"], err["error"][:60])

    # Assemble final markup (no table regions for now - from Phase 1 TSV)
    table_regions = []
    markup = assemble_gazette_markup(results, doc_name, table_regions)

    # Write output
    _log_step(f"Writing output to {output}")
    output.write_text(markup, encoding="utf-8")

    # Write metadata
    metadata = {
        "document_name": doc_name,
        "source_tsv": str(tsv_path),
        "conversion_date": datetime.now().isoformat(),
        "type": "gazette",
        "sections_converted": len(results),
        "sections_failed": len(errors),
    }
    meta_path = output.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Gazette conversion complete")
    logger.info("Markup written to %s", output)
    logger.info("Metadata written to %s", meta_path)

    return str(output)