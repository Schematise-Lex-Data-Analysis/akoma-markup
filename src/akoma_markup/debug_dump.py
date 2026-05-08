"""Sidecar debug-file writers for the conversion pipeline.

Each function writes one inspection-only file under
``<output_dir>/.akoma_debug/`` (auto-created), prefixed with the PDF stem
so multiple PDFs sharing the same output directory don't collide. None of
these feed back into conversion — they exist purely so failures can be
diagnosed without re-running the pipeline.
"""

import csv
import json
import re
from pathlib import Path


_DEBUG_DIR_NAME = ".akoma_debug"


def _debug_path(pdf: Path, output_path: str, name: str) -> Path:
    """Resolve (and create) ``<output_dir>/.akoma_debug/<pdf_stem>_<name>``."""
    debug_dir = Path(output_path).parent / _DEBUG_DIR_NAME
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir / f"{pdf.stem}_{name}"


def write_raw_text(pdf: Path, raw_text: str, output_path: str) -> Path:
    """Dump post-table-rescue text."""
    path = _debug_path(pdf, output_path, "raw_text.txt")
    path.write_text(raw_text)
    return path


def write_ocr_text(pdf: Path, text: str, output_path: str) -> Path:
    """Dump the post-TOC body text (LLM input)."""
    path = _debug_path(pdf, output_path, "ocr.txt")
    path.write_text(text, encoding="utf-8")
    return path


def write_table_regions(
    pdf: Path,
    table_regions: list[dict],
    table_blocks: dict[int, str],
    output_path: str,
) -> Path:
    """Dump per-region OCR markdown + rendered bluebell side-by-side."""
    path = _debug_path(pdf, output_path, "table_regions.txt")
    with open(path, "w") as f:
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
    return path


def write_parser_summary(
    pdf: Path,
    toc_end_line: int,
    section_names: list[dict],
    chapter_ranges: list[dict],
    output_path: str,
) -> Path:
    """Dump TOC structure (chapters → their sections + uncategorized)."""

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

    path = _debug_path(pdf, output_path, "parser_summary.json")
    path.write_text(json.dumps({
        "pdf_file": str(pdf),
        "toc_end_line": toc_end_line,
        "section_count_in_toc": len(section_names),
        "chapter_count": len(chapter_ranges),
        "chapters": chapters_summary,
        "uncategorized_sections": uncategorized,
    }, indent=2, ensure_ascii=False, default=str))
    return path


def write_sections_tsv(
    pdf: Path, sections: list[dict], output_path: str
) -> Path:
    """Append num/heading/content rows to a TSV."""
    path = _debug_path(pdf, output_path, "sections.tsv")
    with open(path, "a") as f:
        writer = csv.writer(f, delimiter="\t")
        for sec in sections:
            content = sec.get("content", "[No content]")
            content_preview = content.replace("\n", " ")
            writer.writerow([sec["num"], sec["heading"], content_preview])
    return path
