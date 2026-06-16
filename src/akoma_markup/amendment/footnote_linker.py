"""Footnote marker extraction and section linking for amendments.

This module links amendment annotations to their target sections
by correlating footnote markers (superscript numbers) found in
the main text with annotations at the bottom of pages.
"""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pdfplumber

from .patterns import (
    FOOTNOTE_ANNOTATION_MARKER_PATTERN,
    FOOTNOTE_MARKER_PATTERN,
    FootnoteContext,
)

if TYPE_CHECKING:
    from .patterns import ExtractedAmendment

logger = logging.getLogger(__name__)

# Unicode superscripts mapping
SUPERSCRIPT_TO_INT = {
    "\u00B9": 1,
    "\u00B2": 2,
    "\u00B3": 3,
    "\u2074": 4,
    "\u2075": 5,
    "\u2076": 6,
    "\u2077": 7,
    "\u2078": 8,
    "\u2079": 9,
}


def _is_likely_section_header(line: str, section_num: str) -> bool:
    """Check if a line that matches section pattern is likely a real section header.
    
    Distinguishes between real section headers (e.g., "1. Short title...") 
    and footnote annotations that happen to start with numbers (e.g., "1. Subs. by...").
    
    Args:
        line: The line to check
        section_num: The section number matched by regex
        
    Returns:
        True if likely a real section header, False if likely a footnote
    """
    import re
    line_lower = line.lower()
    
    # Footnote indicators that mean this is NOT a section header
    footnote_indicators = [
        "subs.",
        "ins.",
        "omitted",
        "vide",
        "notification",
        "g.s.r.",
        "gazette",
        "w.e.f.",
        "(w.e.f.",
    ]
    
    # Check for footnote indicators
    for indicator in footnote_indicators:
        if indicator in line_lower:
            return False
    
    # Real section headers often have certain patterns
    # 1. May have em dash "–" after title
    # 2. Usually don't have dates in parentheses (common in footnotes)
    # 3. Section headers are usually title + optional em dash, not long text
    
    # Check if this looks like a date footnote (e.g., "1. 17th October, 2000...")
    date_patterns = [
        r"\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|"
        r"August|September|October|November|December)",
        r"\d{4},",
        r"\d{1,2}-\d{1,2}-\d{4}",
    ]
    
    for pattern in date_patterns:
        if re.search(pattern, line_lower):
            return False
    
    # If we get here, it's likely a real section header
    return True


def _parse_marker_number(marker: str) -> int:
    """Convert marker string to integer (handles unicode superscripts).

    Args:
        marker: The marker string (e.g., '¹' or '1')

    Returns:
        Integer value of the marker
    """
    if marker in SUPERSCRIPT_TO_INT:
        return SUPERSCRIPT_TO_INT[marker]
    try:
        return int(marker)
    except ValueError:
        return 0


def extract_markers_with_context(
    text: str,
    page_num: int,
    toc_sections: list[dict],
) -> list[FootnoteContext]:
    """Extract footnote markers with their surrounding context.

    Scans page text for superscript or bracketed footnote markers
    and determines which section they likely belong to based on
    surrounding text and proximity to section headers.

    Args:
        text: Page text content
        page_num: Page number for context
        toc_sections: List of section dicts from TOC parsing

    Returns:
        List of FootnoteContext objects with marker info
    """
    contexts: list[FootnoteContext] = []
    lines = text.split("\n")

    # Build set of valid section numbers for validation
    valid_sections = {s["num"] for s in toc_sections}

    for line_num, line in enumerate(lines):
        # Look for footnote markers preceded by text (sentence end)
        matches = list(FOOTNOTE_MARKER_PATTERN.finditer(line))

        for match in matches:
            # Extract marker - group 0 is the pattern, or check for bracketed
            preceding_text = match.group(1) if match.group(1) else ""
            bracketed_num = match.group(2)

            # Determine marker character
            marker_char = None
            if bracketed_num:
                marker_char = bracketed_num
            else:
                # Find the actual superscript character in match
                for char in match.group(0):
                    if char in SUPERSCRIPT_TO_INT:
                        marker_char = char
                        break

            if not marker_char:
                continue

            marker_num = _parse_marker_number(marker_char)

            # Build surrounding context (previous line + this line + next line)
            context_start = max(0, line_num - 1)
            context_end = min(len(lines), line_num + 2)
            surrounding = "\n".join(
                lines[context_start:context_end]
            ).strip()

            # Try to detect section from context
            suspected_section = None

            # Check for section header in preceding lines (current or previous)
            for check_line in reversed(lines[max(0, line_num - 5):line_num + 1]):
                # Match section headers like "10. Heading" or "10A. Heading"
                sec_match = re.match(r"^(\d+[A-Z]?)\.\s+", check_line)
                if sec_match:
                    candidate = sec_match.group(1)
                    if candidate in valid_sections:
                        suspected_section = candidate
                        break

            contexts.append(FootnoteContext(
                marker=marker_char,
                marker_num=marker_num,
                surrounding_text=surrounding,
                suspected_section=suspected_section,
                page_num=page_num,
                line_num=line_num,
            ))

    return contexts


def extract_marker_from_annotation(line: str) -> tuple[str | None, int]:
    """Extract footnote marker from an amendment annotation line.

    Args:
        line: The annotation line (e.g., "¹ Subs. by Act X...")

    Returns:
        Tuple of (marker_string, marker_number) or (None, 0)
    """
    match = FOOTNOTE_ANNOTATION_MARKER_PATTERN.match(line)
    if not match:
        return None, 0

    # Check which group matched: superscript, bracketed, or plain
    marker = match.group(1) or match.group(2) or match.group(3)
    if marker:
        num = _parse_marker_number(marker)
        return marker, num

    return None, 0


def build_page_section_map(
    pdf_path: Path,
    toc_sections: list[dict],
) -> dict[int, list[str]]:
    """Build a map of page numbers to sections appearing on each page.

    Scans through the PDF to build an approximate mapping based on
    where section headers appear. This is used as a fallback when
    footnote marker context is ambiguous.

    Args:
        pdf_path: Path to the PDF file
        toc_sections: List of section dicts from TOC parsing

    Returns:
        Dictionary mapping page_num -> list of section numbers
    """
    page_section_map: dict[int, list[str]] = {}
    valid_sections = {s["num"]: s for s in toc_sections}
    section_order = [s["num"] for s in toc_sections]

    try:
        with pdfplumber.open(pdf_path) as pdf:
            last_seen_section: str | None = None

            for page_num, page in enumerate(pdf.pages, 1):
                sections_on_page: list[str] = []
                text = page.extract_text() or ""
                lines = text.split("\n")

                for line in lines:
                    # Match section headers
                    sec_match = re.match(r"^(\d+[A-Z]?)\.\s+", line.strip())
                    if sec_match:
                        candidate = sec_match.group(1)
                        if candidate in valid_sections and _is_likely_section_header(line.strip(), candidate):
                            sections_on_page.append(candidate)
                            last_seen_section = candidate

                # If no section header on this page but we have a last seen,
                # assume we're mid-section
                if not sections_on_page and last_seen_section:
                    # Check if this might be a continuation
                    # Only add if there's substantial text
                    if len(text.strip()) > 200:
                        sections_on_page = [last_seen_section]

                page_section_map[page_num] = sections_on_page

    except Exception as exc:
        logger.warning(f"Error building page section map: {exc}")

    return page_section_map


def link_footnotes_to_amendments(
    markers: list[FootnoteContext],
    amendments: list["ExtractedAmendment"],
    page_sections: list[str],
    toc_sections: list[dict],
) -> list["ExtractedAmendment"]:
    """Link amendment annotations to sections using footnote markers.

    Uses a priority-based linking strategy:
    1. If amendment has a footnote marker, match it to marker context
    2. If multiple markers on page, use context to determine best match
    3. Fall back to page_sections if no marker context available

    Args:
        markers: Footnote markers found on the page
        amendments: Amendments extracted from the page
        page_sections: Sections known to be on this page
        toc_sections: Full TOC for validation

    Returns:
        Amendments with target_section populated where possible
    """
    # Build marker lookup by number
    marker_by_num: dict[int, FootnoteContext] = {
        m.marker_num: m for m in markers if m.marker_num > 0
    }

    valid_sections = {s["num"] for s in toc_sections}

    for amendment in amendments:
        # If already has section from inline context, validate it
        if amendment.target_section and amendment.target_section in valid_sections:
            amendment.linkage_method = "inline_context"
            amendment.linkage_confidence = "high"
            continue

        # Try to link via footnote marker
        if amendment.footnote_marker:
            marker_num = _parse_marker_number(amendment.footnote_marker)
            if marker_num in marker_by_num:
                ctx = marker_by_num[marker_num]
                if ctx.suspected_section and ctx.suspected_section in valid_sections:
                    amendment.target_section = ctx.suspected_section
                    amendment.linkage_method = "footnote_marker"
                    amendment.linkage_confidence = "high"
                    continue

        # Fall back to page sections
        if len(page_sections) == 1:
            # Only one section on page - high confidence
            amendment.target_section = page_sections[0]
            amendment.linkage_method = "page_inference"
            amendment.linkage_confidence = "medium"
        elif len(page_sections) > 1:
            # Multiple sections - low confidence, pick first
            amendment.target_section = page_sections[0]
            amendment.linkage_method = "page_inference"
            amendment.linkage_confidence = "low"
        else:
            amendment.linkage_method = "unknown"
            amendment.linkage_confidence = "none"

    return amendments


def validate_section_linkages(
    amendments: list["ExtractedAmendment"],
    toc_sections: list[dict],
) -> tuple[list["ExtractedAmendment"], list[str]]:
    """Validate and report on section linkages.

    Args:
        amendments: All extracted amendments
        toc_sections: TOC sections for validation

    Returns:
        Tuple of (validated_amendments, warnings)
    """
    valid_sections = {s["num"] for s in toc_sections}
    warnings: list[str] = []

    linked_count = 0
    unlinked_count = 0
    invalid_count = 0

    for amendment in amendments:
        if not amendment.target_section:
            unlinked_count += 1
        elif amendment.target_section not in valid_sections:
            invalid_count += 1
            warnings.append(
                f"Amendment linked to invalid section "
                f"'{amendment.target_section}'"
            )
        else:
            linked_count += 1

    total = len(amendments)
    if total > 0:
        linkage_rate = linked_count / total * 100
        logger.info(
            f"Section linkage: {linked_count}/{total} ({linkage_rate:.1f}%) "
            f"linked, {unlinked_count} unlinked, {invalid_count} invalid"
        )

    return amendments, warnings
