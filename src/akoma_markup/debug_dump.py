"""Sidecar debug-file writers for the conversion pipeline.

Each function writes one inspection-only file next to the markup output.
None of these feed back into conversion — they exist purely so failures
can be diagnosed without re-running the pipeline.
"""

import csv
import json
import re
from pathlib import Path


def write_raw_text(raw_text: str, output_path: str) -> Path:
    """Dump post-table-rescue text to ``<output>.raw_text_debug.txt``."""
    path = Path(output_path).with_suffix(".raw_text_debug.txt")
    path.write_text(raw_text)
    return path


def write_ocr_text(text: str, output_path: str) -> Path:
    """Dump the post-TOC body text (LLM input) to ``<output>.ocr_debug.txt``."""
    path = Path(output_path).with_suffix(".ocr_debug.txt")
    path.write_text(text, encoding="utf-8")
    return path


def write_table_regions(
    table_regions: list[dict],
    table_blocks: dict[int, str],
    output_path: str,
) -> Path:
    """Dump per-region OCR markdown + rendered bluebell side-by-side."""
    path = Path(output_path).with_suffix(".table_regions_debug.txt")
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
    pdf_path: Path,
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

    path = Path(output_path).with_suffix(".parser_debug.json")
    path.write_text(json.dumps({
        "pdf_file": str(pdf_path),
        "toc_end_line": toc_end_line,
        "section_count_in_toc": len(section_names),
        "chapter_count": len(chapter_ranges),
        "chapters": chapters_summary,
        "uncategorized_sections": uncategorized,
    }, indent=2, ensure_ascii=False, default=str))
    return path


def write_sections_tsv(sections: list[dict], output_path: str) -> Path:
    """Append num/heading/content rows to ``<output>.sections_debug.tsv``."""
    path = Path(output_path).with_suffix(".sections_debug.tsv")
    with open(path, "a") as f:
        writer = csv.writer(f, delimiter="\t")
        for sec in sections:
            content = sec.get("content", "[No content]")
            content_preview = content.replace("\n", " ")
            writer.writerow([sec["num"], sec["heading"], content_preview])
    return path
