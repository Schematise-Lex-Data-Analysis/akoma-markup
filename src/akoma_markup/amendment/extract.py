"""PDF annotation extraction for IndiaCode legislative amendments.

This module provides functions to extract amendment annotations from
IndiaCode PDFs and convert them to structured amendment records.
"""

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pdfplumber

from .patterns import (
    CLAUSE_OMITTED_PATTERN,
    ExtractedAmendment,
    INS_IBID_PATTERN,
    INS_PATTERN,
    OMITTED_IBID_PATTERN,
    OMITTED_PATTERN,
    SECTION_OMITTED_SHORTHAND,
    SUBS_IBID_PATTERN,
    SUBS_PATTERN,
    SUBS_SIMPLE_PATTERN,
    detect_amendment_type,
)
from .registry import AmendmentDetail, AmendmentRecord

logger = logging.getLogger(__name__)


@dataclass
class AmendmentExtractionResult:
    """Result of extracting amendments from a PDF.

    Attributes:
        pdf_path: Path to the source PDF
        amendments: List of extracted amendments
        sections_found: Number of sections processed
        errors: List of any errors encountered
    """

    pdf_path: Path
    amendments: list[ExtractedAmendment]
    sections_found: int
    errors: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "pdf_path": str(self.pdf_path),
            "amendments": [a.to_dict() for a in self.amendments],
            "sections_found": self.sections_found,
            "errors": self.errors,
        }


# ============================================================================
# Main Extraction Functions
# ============================================================================


def extract_amendments_from_pdf(
    pdf_path: str | Path,
    include_context: bool = False,
) -> AmendmentExtractionResult:
    """Extract amendment annotations from an IndiaCode PDF.

    Parses footnotes and annotations to identify amendments made to
    the legislation over time.

    Args:
        pdf_path: Path to the PDF file
        include_context: Whether to include surrounding text context

    Returns:
        AmendmentExtractionResult with extracted amendments
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    amendments: list[ExtractedAmendment] = []
    errors: list[str] = []
    sections_found = 0

    logger.info("Extracting amendments from %s", pdf_path)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Track last amending act for "ibid" references
            last_amending_act: dict[str, str] = {"number": "", "year": ""}

            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    text = page.extract_text() or ""

                    # Extract amendments from this page
                    page_amendments = _extract_from_page_text(
                        text, page_num, last_amending_act
                    )
                    amendments.extend(page_amendments)

                    # Update last amending act if found
                    for amdt in page_amendments:
                        if amdt.act_number and amdt.act_year:
                            last_amending_act["number"] = amdt.act_number
                            last_amending_act["year"] = amdt.act_year

                    # Count sections (rough estimate)
                    sections_found += len(re.findall(r"\n\d+[A-Z]?\.\s", text))

                except Exception as exc:
                    error_msg = f"Error processing page {page_num}: {exc}"
                    logger.warning(error_msg)
                    errors.append(error_msg)

    except Exception as exc:
        error_msg = f"Error opening PDF: {exc}"
        logger.error(error_msg)
        errors.append(error_msg)
        raise

    logger.info(
        "Extracted %d amendments from %s (%d sections found)",
        len(amendments),
        pdf_path,
        sections_found,
    )

    return AmendmentExtractionResult(
        pdf_path=pdf_path,
        amendments=amendments,
        sections_found=sections_found,
        errors=errors,
    )


def _extract_from_page_text(
    text: str,
    page_num: int,
    last_amending_act: dict[str, str],
) -> list[ExtractedAmendment]:
    """Extract amendments from a single page's text.

    Args:
        text: Page text content
        page_num: Page number for logging
        last_amending_act: Dict tracking the last amending act

    Returns:
        List of extracted amendments from this page
    """
    amendments: list[ExtractedAmendment] = []

    # Look for common amendment patterns line by line
    lines = text.split("\n")
    current_section: str | None = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to identify current section from line
        section_match = re.match(r"^(\d+[A-Z]?)\.\s+", line)
        if section_match and not line.startswith(("1.", "2.", "3.", "4.")):
            current_section = section_match.group(1)

        # Skip if this isn't a footnote/reference line
        if not _is_footnote_line(line):
            continue

        # Try to match amendment patterns
        amendment = _match_amendment_pattern(
            line, current_section, last_amending_act
        )
        if amendment:
            amendments.append(amendment)

    return amendments


def _is_footnote_line(line: str) -> bool:
    """Check if a line is likely a footnote/reference line.

    Args:
        line: Line of text to check

    Returns:
        True if line appears to be a footnote
    """
    line_lower = line.lower().strip()

    # Common footnote indicators
    indicators = [
        "subs.",
        "ins.",
        "omitted",
        "clause",
        "[omitted",
        "[repealed",
        # Numbered footnotes like "1. Subs. by..."
        r"^\d+\..*?(?:subs\.|ins\.|omitted)",
    ]

    for indicator in indicators:
        if indicator.startswith("^"):
            if re.search(indicator, line_lower):
                return True
        elif indicator in line_lower:
            return True

    return False


def _match_amendment_pattern(
    line: str,
    current_section: str | None,
    last_amending_act: dict[str, str],
) -> ExtractedAmendment | None:
    """Try to match amendment patterns against a line.

    Args:
        line: Line of text to parse
        current_section: Current section being processed (if known)
        last_amending_act: Dict with last amending act info

    Returns:
        ExtractedAmendment if matched, None otherwise
    """
    line = line.strip()

    # Remove leading footnote numbers (e.g., "1. " or "1. Ins. by...")
    line = re.sub(r"^\d+\.\s*", "", line)

    # Try substitution patterns
    match = SUBS_PATTERN.search(line) or SUBS_SIMPLE_PATTERN.search(line)
    if match:
        groups = match.groups()
        return ExtractedAmendment(
            amendment_type="substitution",
            act_number=groups[0] if len(groups) > 0 else None,
            act_year=groups[1] if len(groups) > 1 else None,
            section_number=groups[2] if len(groups) > 2 else None,
            target_section=current_section,
            original_text=groups[-2] if len(groups) > 2 else None,
            effective_date=groups[-1] if len(groups) > 3 else None,
        )

    # Try ibid substitution
    match = SUBS_IBID_PATTERN.search(line)
    if match and last_amending_act["number"]:
        groups = match.groups()
        return ExtractedAmendment(
            amendment_type="substitution",
            act_number=last_amending_act["number"],
            act_year=last_amending_act["year"],
            section_number=groups[0],
            target_section=current_section,
            original_text=groups[1] if len(groups) > 1 else None,
            effective_date=groups[2] if len(groups) > 2 else None,
            ibid_reference=True,
        )

    # Try insertion patterns
    match = INS_PATTERN.search(line)
    if match:
        groups = match.groups()
        return ExtractedAmendment(
            amendment_type="insertion",
            act_number=groups[0],
            act_year=groups[1],
            section_number=groups[2],
            target_section=current_section,
            effective_date=groups[3] if len(groups) > 3 else None,
        )

    # Try ibid insertion
    match = INS_IBID_PATTERN.search(line)
    if match and last_amending_act["number"]:
        groups = match.groups()
        return ExtractedAmendment(
            amendment_type="insertion",
            act_number=last_amending_act["number"],
            act_year=last_amending_act["year"],
            section_number=groups[0],
            target_section=current_section,
            effective_date=groups[1] if len(groups) > 1 else None,
            ibid_reference=True,
        )

    # Try omission patterns
    match = OMITTED_PATTERN.search(line)
    if match:
        groups = match.groups()
        return ExtractedAmendment(
            amendment_type="deletion",
            act_number=groups[0],
            act_year=groups[1],
            section_number=groups[2],
            target_section=current_section,
            effective_date=groups[3] if len(groups) > 3 else None,
        )

    # Try ibid omission
    match = OMITTED_IBID_PATTERN.search(line)
    if match and last_amending_act["number"]:
        groups = match.groups()
        return ExtractedAmendment(
            amendment_type="deletion",
            act_number=last_amending_act["number"],
            act_year=last_amending_act["year"],
            section_number=groups[0],
            target_section=current_section,
            effective_date=groups[1] if len(groups) > 1 else None,
            ibid_reference=True,
        )

    # Try clause omission pattern
    match = CLAUSE_OMITTED_PATTERN.search(line)
    if match:
        groups = match.groups()
        return ExtractedAmendment(
            amendment_type="deletion",
            act_number=groups[1],
            act_year=groups[2],
            section_number=groups[3],
            target_section=current_section,
            target_clause=groups[0],
            effective_date=groups[4] if len(groups) > 4 else None,
        )

    return None


# ============================================================================
# Registry Generation Functions
# ============================================================================


def generate_registry_csv(
    amendments: list[ExtractedAmendment],
    output_path: str | Path,
    act_name: str = "",
    base_version: str = "",
) -> Path:
    """Generate amendment registry CSV from extracted amendments.

    Args:
        amendments: List of extracted amendments
        output_path: Path to write CSV file
        act_name: Name of the base act
        base_version: Base version identifier

    Returns:
        Path to the written CSV file
    """
    output_path = Path(output_path)

    # Group amendments by act
    amendments_by_act: dict[tuple[str, str], list[ExtractedAmendment]] = {}
    for amdt in amendments:
        key = (amdt.act_number or "Unknown", amdt.act_year or "Unknown")
        if key not in amendments_by_act:
            amendments_by_act[key] = []
        amendments_by_act[key].append(amdt)

    # Write CSV
    fieldnames = [
        "act_name",
        "base_version",
        "amendment_act",
        "amendment_year",
        "gazette_file",
        "sections_amended",
        "effective_date",
        "notes",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (act_num, act_year), act_amendments in amendments_by_act.items():
            sections = sorted(
                set(a.target_section for a in act_amendments if a.target_section)
            )
            effective_dates = sorted(
                set(a.effective_date for a in act_amendments if a.effective_date)
            )

            writer.writerow({
                "act_name": act_name or "Unknown Act",
                "base_version": base_version or "base",
                "amendment_act": f"Act {act_num}",
                "amendment_year": act_year,
                "gazette_file": "",  # Would need additional lookup
                "sections_amended": ",".join(sections) if sections else "",
                "effective_date": effective_dates[0] if effective_dates else "",
                "notes": f"{len(act_amendments)} amendment(s) found",
            })

    logger.info("Generated registry CSV: %s (%d acts)", output_path, len(amendments_by_act))
    return output_path


def generate_details_tsv(
    amendments: list[ExtractedAmendment],
    output_path: str | Path,
) -> Path:
    """Generate amendment details TSV from extracted amendments.

    Args:
        amendments: List of extracted amendments
        output_path: Path to write TSV file

    Returns:
        Path to the written TSV file
    """
    output_path = Path(output_path)

    fieldnames = [
        "section",
        "operation",
        "new_text",
        "effective_date",
        "amendment_act",
        "gazette_ref",
        "target_subsection",
        "target_clause",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for amdt in amendments:
            # Map amendment type to operation
            operation_map = {
                "substitution": "replace",
                "insertion": "insert",
                "deletion": "delete",
            }

            writer.writerow({
                "section": amdt.target_section or "",
                "operation": operation_map.get(amdt.amendment_type, "replace"),
                "new_text": "",  # Would need full text extraction
                "effective_date": amdt.effective_date or "",
                "amendment_act": amdt.amendment_act_id,
                "gazette_ref": "",
                "target_subsection": amdt.target_subsection or "",
                "target_clause": amdt.target_clause or "",
            })

    logger.info("Generated details TSV: %s (%d amendments)", output_path, len(amendments))
    return output_path


# ============================================================================
# Section Context Extractor
# ============================================================================


def extract_section_context(
    pdf_path: str | Path,
    section_number: str,
    context_lines: int = 5,
) -> dict | None:
    """Extract the full text of a section with surrounding context.

    Args:
        pdf_path: Path to the PDF file
        section_number: Section number to extract (e.g., "43" or "43A")
        context_lines: Number of lines of context around the section

    Returns:
        Dict with section text and metadata, or None if not found
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            full_text += page_text + "\n"

    # Find the section
    section_pattern = re.compile(
        rf"(?:^|\n)\s*{re.escape(section_number)}\.\s+(.*?)(?=\n\d+[A-Z]?\.\s|\Z)",
        re.DOTALL,
    )
    match = section_pattern.search(full_text)

    if not match:
        return None

    section_text = match.group(1).strip()

    # Try to extract hierarchy
    subsections = re.findall(r"\n\s*\((\d+)\)\s+([^\n]+)", section_text)
    clauses = re.findall(r"\n\s*\(([a-z])\)\s+([^\n]+)", section_text)

    return {
        "section_number": section_number,
        "section_text": section_text,
        "subsections": [
            {"number": num, "text": text.strip()} for num, text in subsections
        ],
        "clauses": [
            {"letter": let, "text": text.strip()} for let, text in clauses
        ],
    }
