"""TOC and section parsing for legislative documents."""

import difflib
import logging
import re

logger = logging.getLogger(__name__)


def _heading_similarity(heading1: str, heading2: str) -> float:
    """Calculate fuzzy similarity ratio between two headings (0.0 to 1.0)."""
    # Normalize: lowercase, remove punctuation at end, extra spaces
    h1 = heading1.lower().strip().rstrip(".,;:-–—")
    h2 = heading2.lower().strip().rstrip(".,;:-–—")
    return difflib.SequenceMatcher(None, h1, h2).ratio()


def parse_toc(lines: list[str]) -> tuple[list[dict], list[dict], int]:
    """Parse the Table of Contents to extract chapter names and section names.

    Expected TOC format:
        CHAPTER <ROMAN NUMERAL>
        <CHAPTER TITLE IN CAPS>
        <num>. <Section name>.

    Scanning stops when a CHAPTER/PART heading is encountered that was
    already seen earlier in the TOC, which signals the start of body text.
    This works regardless of numeral style (roman, arabic, letter).

    Args:
        lines: All lines from the extracted PDF text.

    Returns:
        Tuple of (chapters, sections, toc_end_line) where toc_end_line is the
        index of the last TOC entry found.
    """
    chapters = []
    sections = []
    toc_end_line = 0
    seen_headings: set[str] = set()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect any CHAPTER or PART line (roman, arabic, letter — any style).
        # If its heading was already seen in the TOC, the body has started.
        if re.match(r"^(CHAPTER|PART)\s+.+$", line) and i + 1 < len(lines):
            next_heading = lines[i + 1].strip()
            if next_heading in seen_headings:
                break

        chapter_match = re.match(r"^CHAPTER\s+([IVXLCA]+)$", line)
        if chapter_match:
            roman = chapter_match.group(1)
            if i + 1 < len(lines):
                title = lines[i + 1].strip()
                if title and title != "SECTIONS" and not title.isdigit():
                    chapters.append({"roman": roman, "heading": title})
                    seen_headings.add(title)
            toc_end_line = i + 1
            i += 2
            continue

        section_match = re.match(r"^(\d+[A-Za-z]*)\.\s+(.+?)\.?\s*$", line)
        if section_match:
            sections.append(
                {"num": section_match.group(1), "heading": section_match.group(2)}
            )
            toc_end_line = i

        i += 1

    validate_toc_sections(sections)
    return chapters, sections, toc_end_line


def validate_toc_sections(sections: list[dict]):
    """Check for gaps or misnumbering in TOC section numbers and log warnings.

    Args:
        sections: Section dicts with 'num' keys from parse_toc.
    """
    nums = []
    for sec in sections:
        m = re.match(r"(\d+)", sec["num"])
        if m:
            nums.append(int(m.group(1)))

    for i in range(1, len(nums)):
        if nums[i] - nums[i - 1] > 1:
            gap_start = nums[i - 1] + 1
            gap_end = nums[i] - 1
            if gap_start == gap_end:
                logger.warning(
                    f"TOC missing section {gap_start} "
                    f"(between {nums[i - 1]} and {nums[i]})"
                )
            else:
                logger.warning(
                    f"TOC missing sections {gap_start}-{gap_end} "
                    f"(between {nums[i - 1]} and {nums[i]})"
                )


def extract_chapter_ranges(
    toc_lines: list[str],
    section_names: list[dict],
    toc_end_line: int | None = None,
) -> list[dict]:
    """Extract chapter ranges (start/end section numbers) from the TOC.

    Args:
        toc_lines: Lines from the TOC portion of the PDF.
        section_names: Section dicts with 'num' and 'heading' keys.
        toc_end_line: If provided, only scan up to this line index.

    Returns:
        List of chapter dicts with 'roman', 'heading', 'start', 'end' keys.
    """
    end = toc_end_line + 1 if toc_end_line is not None else len(toc_lines)
    chapter_ranges = []
    current_chapter = None
    chapter_sections: list[int] = []

    i = 0
    while i < end:
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

        section_match = re.match(r"^(\d+[A-Za-z]*)\.\s+", line)
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
            # Exact match within a chapter range
            for ch in chapter_ranges:
                if ch["start"] <= num <= ch["end"]:
                    return {"roman": ch["roman"], "heading": ch["heading"]}
            # Fallback: assign to nearest chapter by distance
            best = None
            best_dist = float("inf")
            for ch in chapter_ranges:
                dist = min(abs(num - ch["start"]), abs(num - ch["end"]))
                if dist < best_dist:
                    best_dist = dist
                    best = ch
            if best:
                return {"roman": best["roman"], "heading": best["heading"]}
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

    Matches each section by its number (e.g. ``42.``) and captures everything
    until the next section number begins.  Does not rely on heading text or
    dash characters, making it resilient to formatting variations across PDFs.

    Args:
        full_text: Full text of the PDF (post-TOC portion).
        section_names: Section dicts with 'num' and 'heading' keys.

    Returns:
        List of section dicts with 'num', 'heading', and 'content' keys.
    """
    logger.debug(f"Starting section content extraction for {len(section_names)} sections")
    sections_with_content = []

    for i, sec in enumerate(section_names):
        sec_num = sec["num"]
        sec_heading = sec["heading"]

        if i + 1 < len(section_names):
            next_sec_num = section_names[i + 1]["num"]
            stop_pattern = rf"\n{next_sec_num}\."
        else:
            stop_pattern = r"\Z"

        # Find all candidate positions for this section number
        # Pattern: section number followed by text until emdash, newline, or end
        candidate_pattern = rf"\n{re.escape(sec_num)}\.\s+([^\n—–\-]+)"
        candidates = list(re.finditer(candidate_pattern, full_text))

        best_match = None
        best_similarity = 0.0
        for cand in candidates:
            # Extract the heading part from candidate (before emdash/newline)
            cand_heading = cand.group(1).strip()
            # Skip footnotes like "3. Ins. by Act..." or "3. Subs. by..."
            if re.match(r"^(Ins\.|Subs\.|The\.|Certain|A\s|Sub\.|Omitted)", cand_heading, re.IGNORECASE):
                continue
            similarity = _heading_similarity(sec_heading, cand_heading)
            if similarity > best_similarity and similarity >= 0.90:
                best_similarity = similarity
                best_match = cand

        if best_match:
            # Now extract the full content from this position until next section
            start_pos = best_match.end()
            # Find where next section starts
            next_match = re.search(stop_pattern, full_text[start_pos:])
            if next_match:
                content = full_text[start_pos:start_pos + next_match.start()]
            else:
                content = full_text[start_pos:]
        else:
            content = None

        if best_match and content is not None:
            content = content.strip()
            content = re.sub(r"\n\d+\n", "\n", content)
            content = re.sub(r"\d+\[", "[", content)
            logger.debug(f"Section {sec_num} matched with similarity {best_similarity:.2%} - content length: {len(content)} chars")
            sections_with_content.append(
                {"num": sec_num, "heading": sec["heading"], "content": content}
            )
        else:
            fallback = "[Repealed/Omitted]" if (
                "Repealed" in sec["heading"] or "Omitted" in sec["heading"]
            ) else ""
            logger.debug(f"Section {sec_num} - NO CONTENT FOUND (tried {len(candidates)} candidates), using fallback: {fallback!r}")
            sections_with_content.append(
                {"num": sec_num, "heading": sec["heading"], "content": fallback}
            )

    logger.debug(f"Section content extraction complete. Total sections with content: {len([s for s in sections_with_content if s['content']])}")
    return sections_with_content
