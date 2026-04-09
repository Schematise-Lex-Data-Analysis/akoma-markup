"""TOC and section parsing for BNSS 2023 documents."""

import re


def parse_toc(lines: list[str]) -> tuple[list[dict], list[dict]]:
    """Parse the Table of Contents to extract chapter names and section names.

    Expected TOC format:
        CHAPTER <ROMAN NUMERAL>
        <CHAPTER TITLE IN CAPS>
        <num>. <Section name>.

    Args:
        lines: All lines from the extracted PDF text.

    Returns:
        Tuple of (chapters, sections) where each is a list of dicts.
    """
    chapters = []
    sections = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        chapter_match = re.match(r"^CHAPTER\s+([IVXLCA]+)$", line)
        if chapter_match:
            roman = chapter_match.group(1)
            if i + 1 < len(lines):
                title = lines[i + 1].strip()
                if title and title != "SECTIONS" and not title.isdigit():
                    chapters.append({"roman": roman, "heading": title})
            i += 2
            continue

        section_match = re.match(r"^(\d+[A-Za-z]?)\.\s+(.+?)\.?\s*$", line)
        if section_match:
            sections.append(
                {"num": section_match.group(1), "heading": section_match.group(2)}
            )

        i += 1
    print (chapters, sections)
    return chapters, sections


def extract_chapter_ranges(
    toc_lines: list[str], section_names: list[dict]
) -> list[dict]:
    """Extract chapter ranges (start/end section numbers) from the TOC.

    Args:
        toc_lines: Lines from the TOC portion of the PDF.
        section_names: Section dicts with 'num' and 'heading' keys.

    Returns:
        List of chapter dicts with 'roman', 'heading', 'start', 'end' keys.
    """
    chapter_ranges = []
    current_chapter = None
    chapter_sections: list[int] = []

    i = 0
    while i < len(toc_lines):
        line = toc_lines[i].strip()

        chapter_match = re.match(r"^CHAPTER\s+([IVXLCA]+)$", line)
        if chapter_match:
            if current_chapter and chapter_sections:
                current_chapter["start"] = min(chapter_sections)
                current_chapter["end"] = max(chapter_sections)
                chapter_ranges.append(current_chapter)
                chapter_sections = []

            roman = chapter_match.group(1)
            if i + 1 < len(toc_lines):
                heading = toc_lines[i + 1].strip()
                if heading and heading != "SECTIONS" and not heading.isdigit():
                    current_chapter = {"roman": roman, "heading": heading}
            i += 2
            continue

        section_match = re.match(r"^(\d+[A-Za-z]?)\.\s+", line)
        if section_match and current_chapter:
            num_match = re.match(r"(\d+)", section_match.group(1))
            if num_match:
                chapter_sections.append(int(num_match.group(1)))

        i += 1

    if current_chapter and chapter_sections:
        current_chapter["start"] = min(chapter_sections)
        current_chapter["end"] = max(chapter_sections)
        chapter_ranges.append(current_chapter)

    return chapter_ranges


def filter_sections_by_chapters(
    sections: list[dict],
    chapter_ranges: list[dict],
    selected_chapters: list[str] | None = None,
) -> list[dict]:
    """Filter sections to only include specified chapters.

    Args:
        sections: List of section dicts.
        chapter_ranges: List of chapter range dicts.
        selected_chapters: Roman numerals of chapters to include, or None for all.

    Returns:
        Filtered list of sections with chapter info added.
    """

    def _chapter_for_section(section_num: str) -> dict:
        try:
            num_match = re.match(r"(\d+)", section_num)
            if not num_match:
                return {"roman": "NA", "heading": "Unknown"}
            num = int(num_match.group(1))
            for ch in chapter_ranges:
                if ch["start"] <= num <= ch["end"]:
                    return {"roman": ch["roman"], "heading": ch["heading"]}
        except (ValueError, KeyError):
            pass
        return {"roman": "NA", "heading": "Unknown"}

    filtered = []
    for sec in sections:
        chapter = _chapter_for_section(sec["num"])
        if chapter["roman"] == "NA":
            continue
        if selected_chapters is not None and chapter["roman"] not in selected_chapters:
            continue
        sec["chapter_roman"] = chapter["roman"]
        sec["chapter_heading"] = chapter["heading"]
        filtered.append(sec)

    return filtered


def extract_section_content(
    full_text: str, section_names: list[dict]
) -> list[dict]:
    """Extract section content from the main body text.

    Uses section headings from the TOC as regex anchors to locate each
    section's body text.

    Args:
        full_text: Full text of the PDF (post-TOC portion).
        section_names: Section dicts with 'num' and 'heading' keys.

    Returns:
        List of section dicts with 'num', 'heading', and 'content' keys.
    """
    sections_with_content = []

    for i, sec in enumerate(section_names):
        sec_num = sec["num"]
        escaped_heading = re.escape(sec["heading"])

        if i + 1 < len(section_names):
            next_sec_num = section_names[i + 1]["num"]
            stop_pattern = rf"\n{next_sec_num}\."
        else:
            stop_pattern = r"\Z"

        pattern = (
            rf"{sec_num}\.\s+{escaped_heading}[.\s]*[—\-–]\s*(.+?)(?={stop_pattern})"
        )
        match = re.search(pattern, full_text, re.DOTALL)

        if match:
            content = match.group(1).strip()
            content = re.sub(r"\n\d+\n", "\n", content)
            content = re.sub(r"\d+\[", "[", content)
            sections_with_content.append(
                {"num": sec_num, "heading": sec["heading"], "content": content}
            )
        else:
            fallback = "[Repealed/Omitted]" if (
                "Repealed" in sec["heading"] or "Omitted" in sec["heading"]
            ) else ""
            sections_with_content.append(
                {"num": sec_num, "heading": sec["heading"], "content": fallback}
            )

    return sections_with_content
