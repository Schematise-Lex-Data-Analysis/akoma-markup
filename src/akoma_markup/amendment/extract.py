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

from .footnote_linker import (
    build_page_section_map,
    extract_marker_from_annotation,
    extract_markers_with_context,
    link_footnotes_to_amendments,
    validate_section_linkages,
)
from .patterns import (
    CLAUSE_OMITTED_PATTERN,
    ExtractedAmendment,
    INS_IBID_PATTERN,
    INS_PATTERN,
    OMITTED_IBID_PATTERN,
    OMITTED_PATTERN,
    SUBS_IBID_PATTERN,
    SUBS_PATTERN,
    SUBS_SIMPLE_PATTERN,
)

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
    the legislation over time. Uses footnote marker correlation to
    accurately link amendments to their target sections.

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

    # Import here to avoid circular dependency
    from ..parsing.text.chapter_section_mapping import (
        parse_toc,
        preprocess_pdf_text,
    )

    # Pre-scan to get TOC sections
    toc_sections: list[dict] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_text = ""
            for page in pdf.pages[:10]:  # Usually in first 10 pages
                all_text += page.extract_text() or ""
            lines = preprocess_pdf_text(all_text).split("\n")
            _, sections, _ = parse_toc(lines)
            toc_sections = sections
            sections_found = len(sections)
    except Exception as exc:
        logger.warning(f"Could not parse TOC: {exc}")

    logger.info(f"Found {len(toc_sections)} sections in TOC")

    # Build page-to-section map for fallback linking
    page_section_map: dict[int, list[str]] = {}
    try:
        page_section_map = build_page_section_map(pdf_path, toc_sections)
    except Exception as exc:
        logger.warning(f"Could not build page section map: {exc}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Track last amending act for "ibid" references
            last_amending_act: dict[str, str] = {"number": "", "year": ""}

            # Track current section across pages for proper inline context
            current_section: str | None = None
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    text = page.extract_text() or ""

                    # Step 1: Extract footnote markers with context
                    markers = extract_markers_with_context(
                        text, page_num, toc_sections
                    )

                    # Step 2: Extract amendment annotations from this page
                    # Pass current_section to maintain cross-page tracking
                    page_amendments, updated_section = _extract_from_page_text(
                        text, page_num, last_amending_act, current_section
                    )
                    
                    # Update current_section for next page
                    if updated_section is not None:
                        current_section = updated_section

                    # Step 3: Link annotations to sections via footnote markers
                    page_sections = page_section_map.get(page_num, [])
                    linked_amendments = link_footnotes_to_amendments(
                        markers, page_amendments, page_sections, toc_sections
                    )
                    amendments.extend(linked_amendments)

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

    # Validate section linkages and report statistics
    amendments, linkage_warnings = validate_section_linkages(
        amendments, toc_sections
    )
    errors.extend(linkage_warnings)

    # Count linked vs unlinked
    linked = sum(1 for a in amendments if a.target_section)
    logger.info(
        "Extracted %d amendments from %s (%d sections found, "
        "%d linked to sections)",
        len(amendments),
        pdf_path,
        sections_found,
        linked,
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
    current_section: str | None = None,
) -> list[ExtractedAmendment]:
    """Extract amendments from a single page's text.

    Args:
        text: Page text content
        page_num: Page number for logging
        last_amending_act: Dict tracking the last amending act
        current_section: Current section from previous page (for cross-page tracking)

    Returns:
        Tuple of (amendments_list, updated_current_section)
    """
    amendments: list[ExtractedAmendment] = []

    # Look for common amendment patterns line by line
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to identify current section from line
        # Only update if it looks like a real section header, not a footnote
        section_match = re.match(r"^(\d+[A-Z]?)\.\s+", line)
        if section_match:
            # Check if this is likely a real section header vs a footnote annotation
            if _is_likely_section_header(line, section_match.group(1)):
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

    return amendments, current_section


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
    original_line = line.strip()

    # Extract footnote marker (e.g., "¹ " or "[1] " or "1. ")
    marker, _ = extract_marker_from_annotation(original_line)

    # Remove leading footnote numbers (e.g., "1. " or "1. Ins. by...")
    line = re.sub(r"^\d+\.\s*", "", original_line)

    # Try substitution patterns
    match = SUBS_PATTERN.search(line) or SUBS_SIMPLE_PATTERN.search(line)
    if match:
        groups = match.groups()
        
        # Extract target location from pattern groups
        target_subsection = None
        target_clause = None
        
        if len(groups) > 6:
            # SUBS_PATTERN has 9 groups
            # Group 4: sub-section, Group 5: sub-clause, Group 6: clause, Group 7: section
            if groups[3]:  # sub-section
                target_subsection = groups[3]
            elif groups[4]:  # sub-clause
                target_subsection = groups[4]  # Treat sub-clause as subsection
            elif groups[5]:  # clause
                target_clause = groups[5]
            # Group 7 (section) is captured but we already have target_section from context
        
        return ExtractedAmendment(
            amendment_type="substitution",
            act_number=groups[0] if len(groups) > 0 else None,
            act_year=groups[1] if len(groups) > 1 else None,
            section_number=groups[2] if len(groups) > 2 else None,
            target_section=current_section,
            target_subsection=target_subsection,
            target_clause=target_clause,
            original_text=groups[-2] if len(groups) > 2 else None,
            effective_date=groups[-1] if len(groups) > 3 else None,
            footnote_marker=marker,
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
            footnote_marker=marker,
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
            footnote_marker=marker,
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
            footnote_marker=marker,
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
            footnote_marker=marker,
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
            footnote_marker=marker,
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
            footnote_marker=marker,
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

    # Write CSV with per-amendment detail format
    fieldnames = [
        "amendment_type",
        "act_number",
        "act_year",
        "section_number",
        "target_section",
        "target_subsection",
        "target_clause",
        "effective_date",
        "original_text",
        "footnote_marker",
        "linkage_method",
        "linkage_confidence",
        "ibid_reference",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for amdt in amendments:
            writer.writerow({
                "amendment_type": amdt.amendment_type or "",
                "act_number": amdt.act_number or "",
                "act_year": amdt.act_year or "",
                "section_number": amdt.section_number or "",
                "target_section": amdt.target_section or "",
                "target_subsection": amdt.target_subsection or "",
                "target_clause": amdt.target_clause or "",
                "effective_date": amdt.effective_date or "",
                "original_text": amdt.original_text or "",
                "footnote_marker": amdt.footnote_marker or "",
                "linkage_method": amdt.linkage_method or "",
                "linkage_confidence": amdt.linkage_confidence or "",
                "ibid_reference": "TRUE" if amdt.ibid_reference else "FALSE",
            })

    logger.info("Generated registry CSV: %s (%d amendments)",
                output_path, len(amendments))
    return output_path


def generate_details_tsv(
    amendments: list[ExtractedAmendment],
    output_path: str | Path,
    gazette_dir: str | Path | None = None,
    source_pdf: str | Path | None = None,
    extraction_date: str | None = None,
) -> Path:
    """Generate amendment details TSV from extracted amendments.

    Args:
        amendments: List of extracted amendments
        output_path: Path to write TSV file
        gazette_dir: Optional directory containing gazette notification PDFs
            If provided, will attempt to link amendments to gazette references
        source_pdf: Optional source PDF path for metadata
        extraction_date: Optional extraction date for metadata

    Returns:
        Path to the written TSV file
    """
    output_path = Path(output_path)
    
    # Create gazette mapping if gazette_dir provided
    gazette_mapping = {}
    if gazette_dir:
        try:
            from .gazette_registry import link_amendments_to_gazettes
            gazette_mapping = link_amendments_to_gazettes(amendments, Path(gazette_dir))
        except ImportError as e:
            logger.warning(f"Could not import gazette registry: {e}")
        except Exception as e:
            logger.warning(f"Error linking amendments to gazettes: {e}")

    # Map amendment type to operation
    operation_map = {
        "substitution": "replace",
        "insertion": "insert",
        "deletion": "delete",
    }

    fieldnames = [
        "section",
        "operation",
        "new_text",
        "effective_date",
        "amendment_act",
        "gazette_ref",
        "target_subsection",
        "target_clause",
        "footnote_marker",
        "linkage_method",
        "linkage_confidence",
        # Metadata columns
        "source_pdf",
        "extraction_date",
        "validation_status",
        "data_quality_score",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for amdt in amendments:
            # Get gazette reference if available
            gazette_ref = ""
            if gazette_mapping and hasattr(amdt, 'amendment_act_id'):
                gazette_ref = gazette_mapping.get(amdt.amendment_act_id, "")
            
            # Calculate data quality score (simple heuristic)
            data_quality_score = 0.0
            validation_status = "pending"
            
            # Basic validation checks
            validation_issues = []
            
            # Check if amendment has required fields
            if not amdt.target_section:
                validation_issues.append("missing_target_section")
            if not amdt.amendment_act_id:
                validation_issues.append("missing_amendment_act")
            if not amdt.footnote_marker:
                validation_issues.append("missing_footnote_marker")
            
            # Calculate quality score based on available data
            score_factors = 0
            total_factors = 7  # Total factors we check
            
            if amdt.target_section:
                score_factors += 1
            if amdt.amendment_act_id:
                score_factors += 1
            if amdt.footnote_marker:
                score_factors += 1
            if amdt.effective_date:
                score_factors += 1
            if amdt.target_subsection or amdt.target_clause:
                score_factors += 1  # Target location
            if gazette_ref:
                score_factors += 1  # Gazette reference
            if amdt.linkage_confidence == "high":
                score_factors += 1  # High confidence linkage
            
            data_quality_score = round(score_factors / total_factors, 2)
            
            # Set validation status
            if not validation_issues:
                validation_status = "valid"
            else:
                validation_status = f"issues: {','.join(validation_issues)}"
            
            writer.writerow({
                "section": amdt.target_section or "",
                "operation": operation_map.get(amdt.amendment_type, "replace"),
                "new_text": amdt.original_text or "",
                "effective_date": amdt.effective_date or "",
                "amendment_act": amdt.amendment_act_id,
                "gazette_ref": gazette_ref,
                "target_subsection": amdt.target_subsection or "",
                "target_clause": amdt.target_clause or "",
                "footnote_marker": amdt.footnote_marker or "",
                "linkage_method": amdt.linkage_method or "",
                "linkage_confidence": amdt.linkage_confidence or "",
                # Metadata columns
                "source_pdf": str(source_pdf) if source_pdf else "",
                "extraction_date": extraction_date or "",
                "validation_status": validation_status,
                "data_quality_score": data_quality_score,
            })

    logger.info("Generated details TSV: %s (%d amendments)",
                output_path, len(amendments))
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


async def extract_amendments_hybrid(
    pdf_path: str | Path,
    config: dict | None = None,
) -> AmendmentExtractionResult:
    """Extract amendments using hybrid vision+regex approach.
    
    Combines regex-based text extraction with vision LLM analysis
    for improved accuracy, especially on complex layouts and scanned PDFs.
    
    Args:
        pdf_path: Path to the PDF file
        config: Configuration for hybrid extraction
        
    Returns:
        AmendmentExtractionResult with extracted amendments
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    logger.info(f"Starting hybrid extraction from {pdf_path}")
    
    try:
        # Import here to avoid circular dependency
        from .hybrid_extractor import (
            HybridAmendmentExtractor,
            HybridExtractionConfig
        )
        
        # Create hybrid extractor with config
        hybrid_config = HybridExtractionConfig(
            use_vision=config.get("use_vision", True) if config else True,
            use_regex=config.get("use_regex", True) if config else True,
            confidence_threshold=config.get("confidence_threshold", 0.5) if config else 0.5,
            prefer_vision=config.get("prefer_vision", True) if config else True,
            require_agreement=config.get("require_agreement", False) if config else False
        )
        
        extractor = HybridAmendmentExtractor(hybrid_config)
        
        # Run hybrid extraction
        hybrid_result = await extractor.extract(pdf_path)
        
        # Convert hybrid result to AmendmentExtractionResult format
        amendments = []
        errors = []
        
        for amendment in hybrid_result.amendments:
            try:
                # Convert to ExtractedAmendment format
                # This is a simplified conversion - in practice would need
                # to handle different amendment formats
                amendments.append(amendment)
            except Exception as e:
                errors.append(f"Failed to convert amendment: {e}")
        
        logger.info(f"Hybrid extraction complete: {len(amendments)} amendments")
        
        return AmendmentExtractionResult(
            pdf_path=pdf_path,
            amendments=amendments,
            sections_found=0,  # Would need to extract from hybrid_result
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Hybrid extraction failed: {e}")
        # Fall back to regex-only extraction
        logger.info("Falling back to regex-only extraction")
        return extract_amendments_from_pdf(pdf_path)
