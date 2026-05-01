"""akoma-markup: Convert legislative PDFs to Akoma Ntoso markup."""

import json
import re
import sys
from pathlib import Path

from .conversion import build_chain, process_all_sections
from .util.pdf.text import extract_pdf_pages, extract_pdf_text
from .util.llm.factory import build_llm
from .parsing.tables.render import render_region
from .parsing.text.chapter_section_mapping import (
    extract_chapter_ranges,
    extract_section_content,
    filter_sections_by_chapters,
    parse_toc,
)
from .output import write_markup, write_metadata, write_ocr_text


# Recognises the rendered ``<<TABLE_REGION:N>>`` sentinel on its own line
# (with arbitrary indentation). Used during splice to read the indent and
# replace the line with a re-indented bluebell TABLE block.
_SENTINEL_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)<<TABLE_REGION:(?P<id>\d+)>>[ \t]*$",
    re.MULTILINE,
)


def _splice_sentinels(
    markup: str, table_blocks: dict[int, str]
) -> tuple[str, set[int]]:
    """Replace each ``<<TABLE_REGION:N>>`` sentinel line with its TABLE block.

    The replacement block is re-indented to match the sentinel's leading
    whitespace, so a sentinel sitting under SUBSEC (4-space indent) yields a
    TABLE block whose ``TABLE`` keyword is also at 4-space indent and whose
    children nest from there. Returns the rewritten markup and the set of
    region IDs that were consumed (so the caller can identify trailing
    regions that need to be emitted as standalone TABLE blocks).
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


def convert(
    pdf_path: str,
    llm_config: dict,
    output_path: str | None = None,
    document_name: str | None = None,
    act_number: str | None = None,
    replaces: str | None = None,
    table_mode: str | None = None,
    table_pages: list[int] | None = None,
    azure_ocr_key: str | None = None,
    azure_ocr_endpoint: str | None = None,
    azure_ocr_model: str | None = None,
    azure_vision_key: str | None = None,
    azure_vision_endpoint: str | None = None,
    azure_vision_model: str | None = None,
    azure_vision_api_style: str | None = None,
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
            pages are sent to Azure OCR and converted to Laws.Africa TABLE
            blocks. Requires the three ``azure_ocr_*`` arguments.
            "auto" additionally requires the ``azure_vision_*`` arguments
            (or the matching env vars) for the vision-LLM page classifier.
        table_pages: 1-indexed page list. Required when `table_mode="declared"`.
        azure_ocr_key: Azure OCR API key. Required when `table_mode` is set.
        azure_ocr_endpoint: Azure OCR endpoint. Required when `table_mode` is set.
        azure_ocr_model: Azure OCR model name. Required when `table_mode` is set.
        azure_vision_key: Vision-LLM API key (auto mode).
        azure_vision_endpoint: Vision-LLM endpoint (auto mode).
        azure_vision_model: Vision-LLM model/deployment name (auto mode).
        azure_vision_api_style: Vision-LLM API style — one of 'chat',
            'responses', 'azure-inference' (auto mode).

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
            raise ValueError(
                f"table_mode must be 'declared', 'auto', or 'full'; "
                f"got {table_mode!r}"
            )
        if not azure_ocr_key:
            raise ValueError("table_mode requires azure_ocr_key")
        if not azure_ocr_endpoint:
            raise ValueError("table_mode requires azure_ocr_endpoint")
        if not azure_ocr_model:
            raise ValueError("table_mode requires azure_ocr_model")
        if table_mode == "declared" and not table_pages:
            raise ValueError("table_mode='declared' requires table_pages")
        if table_mode == "auto":
            if not (azure_vision_key and azure_vision_endpoint
                    and azure_vision_model and azure_vision_api_style):
                raise ValueError(
                    "table_mode='auto' requires azure_vision_key, "
                    "azure_vision_endpoint, azure_vision_model, and "
                    "azure_vision_api_style"
                )

    llm = build_llm(llm_config)

    print("Extracting text from PDF ...", file=sys.stderr)
    per_page_text = extract_pdf_pages(str(pdf))

    table_regions: list[dict] = []
    table_blocks: dict[int, str] = {}
    if table_mode is not None:
        from .parsing.tables.rescue import rescue_tables
        print(
            f"Rescuing tables via Azure OCR (mode={table_mode!r}) ...",
            file=sys.stderr,
        )
        per_page_text, table_regions = rescue_tables(
            pdf_path=pdf,
            per_page_text=per_page_text,
            mode=table_mode,
            azure_ocr_key=azure_ocr_key,
            azure_ocr_endpoint=azure_ocr_endpoint,
            azure_ocr_model=azure_ocr_model,
            table_pages=table_pages,
            azure_vision_key=azure_vision_key,
            azure_vision_endpoint=azure_vision_endpoint,
            azure_vision_model=azure_vision_model,
            azure_vision_api_style=azure_vision_api_style,
        )
        # Pre-render every region to a bluebell TABLE block (deterministic,
        # no LLM). The result is held in memory and spliced into the section
        # LLM's output at sentinel positions later.
        table_blocks = {r["id"]: render_region(r["markdown"]) for r in table_regions}
        print(
            f"[tables] rescued {len(table_regions)} region(s); "
            f"rendered {len(table_blocks)} bluebell TABLE blocks",
            file=sys.stderr,
        )
        if table_regions:
            regions_debug_path = Path(output_path or pdf.with_name(
                f"{pdf.stem}_markup.txt"
            )).with_suffix(".table_regions_debug.txt")
            with open(regions_debug_path, "w") as f:
                for r in table_regions:
                    pages = r["pages"]
                    span = (
                        f"page {pages[0]}"
                        if len(pages) == 1
                        else f"pages {pages[0]}-{pages[-1]}"
                    )
                    f.write(f"{'=' * 80}\nREGION {r['id']}  ({span})\n{'=' * 80}\n\n")
                    f.write("--- raw OCR markdown ---\n")
                    f.write(r["markdown"])
                    f.write("\n\n--- rendered bluebell ---\n")
                    f.write(table_blocks.get(r["id"], "(none)"))
                    f.write("\n\n")
            print(
                f"Table-region debug written to {regions_debug_path}",
                file=sys.stderr,
            )

    raw_text = "\n".join(per_page_text)

    raw_text_debug_path = Path(output_path).with_suffix(".raw_text_debug.txt")
    raw_text_debug_path.write_text(raw_text)
    print(f"Raw text (post-rescue) written to {raw_text_debug_path}", file=sys.stderr)

    all_lines = raw_text.splitlines()

    print("Parsing table of contents ...", file=sys.stderr)
    _chapters, section_names, toc_end_line = parse_toc(all_lines)
    chapter_ranges = extract_chapter_ranges(all_lines, section_names, toc_end_line)
    print(
        f"Found {len(chapter_ranges)} chapters, {len(section_names)} sections "
        f"(TOC ends at line {toc_end_line})",
        file=sys.stderr,
    )
    print("[parser] chapters detected:", file=sys.stderr)
    for ch in _chapters:
        print(f"  - {ch.get('roman', '?'):>6}  {ch.get('heading', '?')}",
              file=sys.stderr)
    print("[parser] chapter ranges:", file=sys.stderr)
    for ch in chapter_ranges:
        print(
            f"  - CHAPTER {ch.get('roman', '?'):>6}  "
            f"sections {ch.get('start', '?')}-{ch.get('end', '?')}  "
            f"{ch.get('heading', '?')}",
            file=sys.stderr,
        )

    print(f"OCR text (raw): {all_lines[toc_end_line + 1:][:10]}", file=sys.stderr)
    content_text = "\n".join(all_lines[toc_end_line + 1:])
    sections = extract_section_content(content_text, section_names)

    sections = filter_sections_by_chapters(sections, chapter_ranges)

    # Superficial parser-output dump: TOC structure with each chapter
    # showing the sections it covers. Useful for spotting wrong TOC parsing
    # or misaligned chapter ranges before blaming the LLM.
    def _section_num_int(num_str: str) -> int | None:
        m = re.match(r"(\d+)", num_str or "")
        return int(m.group(1)) if m else None

    chapters_summary: list[dict] = []
    assigned_section_nums: set[str] = set()
    for ch in chapter_ranges:
        ch_sections = []
        for sec in section_names:
            n = _section_num_int(sec["num"])
            if (
                n is not None
                and ch.get("start") is not None
                and ch.get("end") is not None
                and ch["start"] <= n <= ch["end"]
            ):
                ch_sections.append(
                    {"num": sec["num"], "heading": sec["heading"]}
                )
                assigned_section_nums.add(sec["num"])
        chapters_summary.append({
            "roman": ch.get("roman"),
            "heading": ch.get("heading"),
            "section_range": [ch.get("start"), ch.get("end")],
            "section_count": len(ch_sections),
            "sections": ch_sections,
        })

    uncategorized = [
        {"num": s["num"], "heading": s["heading"]}
        for s in section_names
        if s["num"] not in assigned_section_nums
    ]

    parser_debug_path = Path(output_path).with_suffix(".parser_debug.json")
    parser_debug_path.write_text(json.dumps({
        "pdf_file": str(pdf),
        "toc_end_line": toc_end_line,
        "section_count_in_toc": len(section_names),
        "chapter_count": len(chapter_ranges),
        "chapters": chapters_summary,
        "uncategorized_sections": uncategorized,
    }, indent=2, ensure_ascii=False, default=str))
    print(f"Parser debug written to {parser_debug_path}", file=sys.stderr)

    debug_tsv_path = Path(output_path).with_suffix(".sections_debug.tsv")
    with open(debug_tsv_path, 'a') as debug_file:
        import csv
        writer = csv.writer(debug_file, delimiter='\t')
        for sec in sections:
            content = sec.get('content', '[No content]')
            content_preview = content.replace('\n', ' ')
            writer.writerow([sec['num'], sec['heading'], content_preview])
            print(f"Section debug: number={sec['num']}, heading={sec['heading']}", file=sys.stderr)
    seen = set()
    unique = []
    for sec in sections:
        if sec["num"] not in seen:
            seen.add(sec["num"])
            unique.append(sec)
    sections = unique
    print(f"{len(sections)} unique sections ready for conversion", file=sys.stderr)

    ocr_path = write_ocr_text(content_text, output_path)
    print(f"OCR text written to {ocr_path}", file=sys.stderr)

    chain = build_chain(llm, document_name=document_name)
    checkpoint_dir = Path(output_path).parent / ".akoma_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_filename = f"{pdf.stem}_conversion_checkpoint.json"
    checkpoint_path = checkpoint_dir / checkpoint_filename
    converted, errors = process_all_sections(
        chain, sections, checkpoint_path=checkpoint_path
    )

    sec_lookup = {s["num"]: s for s in sections}
    for conv in converted:
        orig = sec_lookup.get(conv["num"], {})
        conv["chapter_roman"] = orig.get("chapter_roman", "NA")
        conv["chapter_heading"] = orig.get("chapter_heading", "Unknown")
        conv["kind"] = "section"

    # The LLM is told to copy <<TABLE_REGION:N>> sentinels verbatim; here we
    # swap them for the deterministically-rendered bluebell TABLE blocks so
    # table data never passes through the LLM's drop/truncation tax.
    consumed_region_ids: set[int] = set()
    if table_blocks:
        for conv in converted:
            spliced, consumed = _splice_sentinels(conv["markup"], table_blocks)
            conv["markup"] = spliced
            consumed_region_ids |= consumed
        print(
            f"[tables] spliced {len(consumed_region_ids)}/{len(table_blocks)} "
            f"region(s) into section markup",
            file=sys.stderr,
        )

    # Regions whose sentinel fell outside any TOC-listed section are
    # emitted as standalone top-level TABLE blocks — no wrapper, no label.
    trailing_regions = [
        r for r in table_regions if r["id"] not in consumed_region_ids
    ]
    if trailing_regions:
        print(
            f"[tables] {len(trailing_regions)} region(s) had no enclosing "
            f"section; emitting as top-level TABLE block(s)",
            file=sys.stderr,
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

    ocr_path = write_ocr_text(content_text, output_path)
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

