"""TOC and section parsing for legislative documents."""

import difflib
import logging
import re

logger = logging.getLogger(__name__)


def preprocess_pdf_text(text: str) -> str:
    """Pre-process PDF text to remove page break artifacts.

    Removes common patterns that appear due to page breaks:
    - Standalone page numbers (lines with just digits)
    - Common headers like "THE GAZETTE OF INDIA", "EXTRAORDINARY", etc.
    - Page number artifacts like "3[" which should be "["

    Args:
        text: Raw text extracted from PDF.

    Returns:
        Cleaned text with page artifacts removed.
    """
    # Remove lines that are just page numbers (standalone digits)
    # This handles cases where page numbers appear on their own line
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # Remove common legislative document headers
    header_patterns = [
        r'THE GAZETTE OF INDIA\s*\n',
        r'EXTRAORDINARY\s*\n',
        r'PART \w+[-–—]\w+\s*\n',
        r'Section \w+\s*\n',
        r'\[.*\d{4}\]\s*\n',  # Lines like [No. 22 of 2021]
    ]
    for pattern in header_patterns:
        text = re.sub(pattern, '\n', text, flags=re.IGNORECASE)

    # Normalize multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


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

        section_match = re.match(r"^(\d+[A-Za-z]*)\.\s+(.+)$", line)
        if section_match:
            sec_num = section_match.group(1)
            sec_heading = section_match.group(2).strip()

            # Handle multi-line titles: check if next line continues the heading
            # (doesn't start with a number pattern or CHAPTER/PART)
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                # Stop if next line looks like a new section, chapter, or is empty
                if not next_line:
                    break
                if re.match(r"^\d+[A-Za-z]*\.\s+", next_line):  # New section
                    break
                if re.match(r"^(CHAPTER|PART)\s+", next_line, re.IGNORECASE):
                    break
                # If it's all caps, might be a chapter title
                if next_line.isupper() and len(next_line) > 10:
                    break
                # Otherwise, append to heading (with space)
                sec_heading += " " + next_line
                j += 1

            # Clean up heading: remove trailing period if present
            sec_heading = sec_heading.rstrip('.')
            sections.append({"num": sec_num, "heading": sec_heading})
            toc_end_line = j - 1 if j > i + 1 else i

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


def _find_section_boundary(text: str, sec_num: str, sec_heading: str) -> tuple[int | None, int | None]:
    """Find the start and end positions of a section using string-based fencing.

    Uses fuzzy matching to identify the correct section header, then fences
    the content between this section and the next one. Now also supports
    heading-based matching when section numbers are ambiguous or missing.

    Args:
        text: Full text to search in.
        sec_num: Section number to find (e.g., "11", "67A").
        sec_heading: Expected heading for validation.

    Returns:
        Tuple of (start_pos, end_pos) where content should be extracted from,
        or (None, None) if not found.
    """
    # Pattern to find section number followed by heading text
    # More flexible: handles various dash types and spacing
    patterns = [
        rf"\n{re.escape(sec_num)}\.\s*([^\n—–\-]*)",  # Until newline or dash
        rf"\n{re.escape(sec_num)}\.\s*([^\n]*)",       # Until just newline
    ]

    candidates = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            cand_heading = match.group(1).strip()
            # Skip footnotes
            if re.match(r"^(Ins\.|Subs\.|The\.|Certain|A\s|Sub\.|Omitted)", cand_heading, re.IGNORECASE):
                continue
            similarity = _heading_similarity(sec_heading, cand_heading)
            candidates.append((match, similarity))

    # Check if we have good number-based matches
    best_num_similarity = max((sim for _, sim in candidates), default=0.0)

    # If no good number-based matches, try heading-based matching
    if not candidates or best_num_similarity < 0.8:
        logger.debug(f"Section {sec_num}: Trying heading-based fallback for '{sec_heading[:50]}...'")

        # Search for the heading text on its own line (possibly with section number)
        # This catches cases like "Definitions" or "1. Definitions" or "1 Definitions"
        heading_variants = [
            rf"\n{re.escape(sec_heading)}\s*\n",  # Heading on its own line
            rf"\n{re.escape(sec_heading)}\s*[—–\-]",  # Heading followed by dash
            rf"\n\d+\.?\s*{re.escape(sec_heading)}\s*\n",  # Number + heading variant
        ]

        for pattern in heading_variants:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matched_text = match.group(0).strip()
                # Extract just the heading part for comparison
                heading_match = re.search(r"[A-Za-z].*", matched_text)
                if heading_match:
                    cand_heading = heading_match.group(0).strip()
                    similarity = _heading_similarity(sec_heading, cand_heading)
                    if similarity >= 0.9:  # High threshold for heading-only matches
                        logger.debug(f"Section {sec_num}: Found via heading match (sim={similarity:.2f})")
                        return match.start(), match.end()

    if not candidates:
        return None, None

    # Sort by similarity and return best match
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_match, best_similarity = candidates[0]

    if best_similarity < 0.90:
        return None, None

    return best_match.start(), best_match.end()


def extract_section_content(
    full_text: str, section_names: list[dict]
) -> list[dict]:
    """Extract section content from the main body text using string-based fencing.

    Uses the ToC's ordered section list to fence boundaries reliably by
    searching for literal section number strings in the text. This handles
    page breaks better than pure regex approaches.

    Args:
        full_text: Full text of the PDF (post-TOC portion).
        section_names: Section dicts with 'num' and 'heading' keys.

    Returns:
        List of section dicts with 'num', 'heading', and 'content' keys.
    """
    logger.debug(f"Starting section content extraction for {len(section_names)} sections")

    # Pre-process text to handle page artifacts
    full_text = preprocess_pdf_text(full_text)

    sections_with_content = []

    for i, sec in enumerate(section_names):
        sec_num = sec["num"]
        sec_heading = sec["heading"]

        # Find the start of this section
        start_match_pos, _ = _find_section_boundary(full_text, sec_num, sec_heading)

        if start_match_pos is None:
            # Try fuzzy fallback
            logger.debug(f"Section {sec_num}: Primary search failed, trying fallback")
            fallback_pattern = rf"\n{re.escape(sec_num)}\.\s+(.+?)(?:\n|$)"
            fallback_match = re.search(fallback_pattern, full_text)
            if fallback_match:
                start_match_pos = fallback_match.start()
                logger.debug(f"Section {sec_num}: Found via fallback")

        if start_match_pos is None:
            # Section not found
            fallback = "[Repealed/Omitted]" if (
                "Repealed" in sec["heading"] or "Omitted" in sec["heading"]
            ) else ""
            logger.debug(f"Section {sec_num}: NO CONTENT FOUND, using fallback: {fallback!r}")
            sections_with_content.append(
                {"num": sec_num, "heading": sec["heading"], "content": fallback}
            )
            continue

        # Find the end: search for the next section's number in the text
        # Use string-based fencing - literally search for next section number
        end_pos = None
        if i + 1 < len(section_names):
            next_sec_num = section_names[i + 1]["num"]
            next_sec_heading = section_names[i + 1]["heading"]

            # Search for next section starting from current section's position
            search_start = start_match_pos + len(sec_num) + 2  # Skip past current section header
            remaining_text = full_text[search_start:]

            # Try to find next section using the same boundary detection
            next_start_rel, _ = _find_section_boundary(remaining_text, next_sec_num, next_sec_heading)

            if next_start_rel is not None:
                end_pos = search_start + next_start_rel
            else:
                # Fallback: simple pattern match for next section
                next_pattern = rf"\n{re.escape(next_sec_num)}\.\s"
                next_match = re.search(next_pattern, remaining_text)
                if next_match:
                    end_pos = search_start + next_match.start()

        # Extract content
        if end_pos is not None:
            content = full_text[start_match_pos:end_pos]
        else:
            content = full_text[start_match_pos:]

        # Clean up content
        content = content.strip()

        # Remove section header line from content (keep only the body)
        content_lines = content.split('\n')
        if len(content_lines) > 1:
            # First line is the header, rest is content
            content = '\n'.join(content_lines[1:]).strip()

        # Final cleanup
        content = re.sub(r"\n\d+\n", "\n", content)
        content = re.sub(r"\d+\[", "[", content)

        logger.debug(f"Section {sec_num}: Extracted {len(content)} chars")
        sections_with_content.append(
            {"num": sec_num, "heading": sec["heading"], "content": content}
        )

    logger.debug(f"Section content extraction complete. Total sections with content: {len([s for s in sections_with_content if s['content']])}")
    return sections_with_content
