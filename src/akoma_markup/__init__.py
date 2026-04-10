"""akoma-markup: Convert BNSS 2023 PDF to Akoma Ntoso markup."""

import sys
from pathlib import Path

from .converter import build_chain, process_all_sections
from .extractor import extract_pdf_text
from .llm import build_llm
from .parser import (
    extract_chapter_ranges,
    extract_section_content,
    filter_sections_by_chapters,
    parse_toc,
)
from .writer import write_markup, write_metadata


def convert(
    pdf_path: str,
    llm_config: dict,
    output_path: str | None = None,
) -> str:
    """Convert a BNSS 2023 PDF to Akoma Ntoso markup.

    Args:
        pdf_path: Path to the BNSS PDF file.
        llm_config: LLM provider config dict. Must include 'provider' key.
            Example: {"provider": "openai", "model": "gpt-4o", "api_key": "sk-..."}
        output_path: Destination for the markup file.
            Defaults to ``<pdf_stem>_markup.txt`` in the same directory.

    Returns:
        Path to the generated markup file.
    """
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    if output_path is None:
        output_path = str(pdf.with_name(f"{pdf.stem}_markup.txt"))

    llm = build_llm(llm_config)

    # 1. Extract text
    print("Extracting text from PDF ...", file=sys.stderr)
    raw_text = extract_pdf_text(str(pdf))
    all_lines = raw_text.splitlines()

    # 2. Parse TOC
    print("Parsing table of contents ...", file=sys.stderr)
    _chapters, section_names, toc_end_line = parse_toc(all_lines)
    chapter_ranges = extract_chapter_ranges(all_lines, section_names, toc_end_line)
    print(
        f"Found {len(chapter_ranges)} chapters, {len(section_names)} sections "
        f"(TOC ends at line {toc_end_line})",
        file=sys.stderr,
    )

    # 3. Extract section content (skip TOC)
    content_text = "\n".join(all_lines[toc_end_line + 1:])
    sections = extract_section_content(content_text, section_names)

    # 4. Map sections to chapters and deduplicate
    sections = filter_sections_by_chapters(sections, chapter_ranges)
    seen = set()
    unique = []
    for sec in sections:
        if sec["num"] not in seen:
            seen.add(sec["num"])
            unique.append(sec)
    sections = unique
    print(f"{len(sections)} unique sections ready for conversion", file=sys.stderr)

    # 5. Convert via LLM
    chain = build_chain(llm)
    checkpoint_dir = Path(output_path).parent / ".akoma_checkpoints"
    converted, errors = process_all_sections(
        chain, sections, checkpoint_dir=checkpoint_dir
    )

    # Merge chapter info
    sec_lookup = {s["num"]: s for s in sections}
    for conv in converted:
        orig = sec_lookup.get(conv["num"], {})
        conv["chapter_roman"] = orig.get("chapter_roman", "NA")
        conv["chapter_heading"] = orig.get("chapter_heading", "Unknown")

    # 6. Write output
    markup_path = write_markup(converted, output_path)
    meta_path = write_metadata(converted, errors, output_path)

    print(f"Markup written to {markup_path}", file=sys.stderr)
    print(f"Metadata written to {meta_path}", file=sys.stderr)
    if errors:
        print(f"{len(errors)} sections failed conversion", file=sys.stderr)

    return markup_path
