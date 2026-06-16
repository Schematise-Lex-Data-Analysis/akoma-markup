"""Regex patterns for extracting amendment annotations from IndiaCode PDFs.

This module provides patterns for parsing amendment footnotes found in
IndiaCode legislative PDFs. The patterns cover common amendment operations:
- Substitution (Subs.): Replace text with new text
- Insertion (Ins.): Add new content
- Omission/Omitted (Omitted): Remove content
"""

import re
from dataclasses import dataclass
from typing import Pattern


# ============================================================================
# Amendment Operation Type Patterns
# ============================================================================

# Substitution patterns - "Subs. by Act X of YYYY, s.Z, for ... (w.e.f. DATE)"
SUBS_PATTERN = re.compile(
    r"Subs\.\s+by\s+Act\s+(\d+)\s+of\s+(\d{4}),\s*s\.\s*(\d+),\s*"
    r"(?:for\s+)?(?:sub-section\s*\(?([^)]+)\)?|sub-clause\s*\(?([^)]+)\)?|"
    r"clause\s*\(?([^)]+)\)?|section\s*\(?([^)]+)\)?|)"
    r"(?:,\s*for\s+)?['\"\u2018\u201c]?(.*?)['\"\u2019\u201d]?\s*"
    r"(?:\(w\.e\.f\.\s+([^)]+)\))?",
    re.IGNORECASE
)

# Simplified substitution pattern for common cases
SUBS_SIMPLE_PATTERN = re.compile(
    r"Subs\.\s+by\s+Act\s+(\d+)\s+of\s+(\d{4}),\s*s\.\s*(\d+),\s*"
    r"for\s+['\"\u2018\u201c]?(.*?)['\"\u2019\u201d]?\s*"
    r"(?:\(w\.e\.f\.\s+([^)]+)\))?",
    re.IGNORECASE
)

# "Subs. by s. X, ibid." pattern - referring to same act as previous
SUBS_IBID_PATTERN = re.compile(
    r"Subs\.\s+by\s+s\.\s*(\d+),\s*ibid\.?,\s*"
    r"for\s+['\"\u2018\u201c]?(.*?)['\"\u2019\u201d]?\s*"
    r"(?:\(w\.e\.f\.\s+([^)]+)\))?",
    re.IGNORECASE
)

# Insertion patterns - "Ins. by Act X of YYYY, s.Z (w.e.f. DATE)"
INS_PATTERN = re.compile(
    r"Ins\.\s+by\s+Act\s+(\d+)\s+of\s+(\d{4}),\s*s\.\s*(\d+)\s*"
    r"(?:\(w\.e\.f\.\s+([^)]+)\))?",
    re.IGNORECASE
)

# "Ins. by s. X, ibid." pattern - referring to same act as previous
INS_IBID_PATTERN = re.compile(
    r"Ins\.\s+by\s+s\.\s*(\d+),\s*ibid\.?\s*"
    r"(?:\(w\.e\.f\.\s+([^)]+)\))?",
    re.IGNORECASE
)

# Omission patterns - "Omitted by Act X of YYYY, s.Z (w.e.f. DATE)"
OMITTED_PATTERN = re.compile(
    r"Omitted\s+by\s+Act\s+(\d+)\s+of\s+(\d{4}),\s*s\.\s*(\d+)\s*"
    r"(?:\(w\.e\.f\.\s+([^)]+)\))?",
    re.IGNORECASE
)

# "Omitted by s. X, ibid." pattern
OMITTED_IBID_PATTERN = re.compile(
    r"Omitted\s+by\s+s\.\s*(\d+),\s*ibid\.?\s*"
    r"(?:\(w\.e\.f\.\s+([^)]+)\))?",
    re.IGNORECASE
)

# Clause omission pattern - "Clauses (a), (d) and (f) omitted by..."
CLAUSE_OMITTED_PATTERN = re.compile(
    r"Clause(?:s)?\s+([^)]+)\s+omitted\s+by\s+Act\s+(\d+)\s+of\s+(\d{4}),\s*s\.\s*(\d+)\s*"
    r"(?:\(w\.e\.f\.\s+([^)]+)\))?",
    re.IGNORECASE
)

# Section omitted shorthand - "Section [Omitted.]"
SECTION_OMITTED_SHORTHAND = re.compile(
    r"^\s*(\d+[A-Z]?)\.\s*\[?\s*Omitted\s*\.?\]?\s*\.?$",
    re.IGNORECASE
)

# ============================================================================
# Context Patterns
# ============================================================================

# Pattern to find section reference preceding an amendment note
SECTION_CONTEXT_PATTERN = re.compile(
    r"(?:^|\n)\s*(\d+[A-Z]?)\.\s*(.+?)(?=\n\d+\.|$)",
    re.DOTALL
)

# Pattern to find sub-section/clause references
SUBSECTION_REF_PATTERN = re.compile(r"\((\d+)\)")
CLAUSE_REF_PATTERN = re.compile(r"\(([a-z])\)")
SUBCLAUSE_REF_PATTERN = re.compile(r"\(([i-v]+)\)")

# ============================================================================
# Footnote Marker Patterns
# ============================================================================

# Unicode superscripts: ¹, ², ³, ⁴-⁹
UNICODE_SUPERSCRIPTS = "\u00B9\u00B2\u00B3\u2074\u2075\u2076\u2077\u2078\u2079"

# Superscript footnote markers appearing inline with text
FOOTNOTE_MARKER_PATTERN = re.compile(
    rf"([^.!?\n]*[.!?])\s*(?:[{UNICODE_SUPERSCRIPTS}]|\[(\d+)\])(?=\s|$|\n)",
    re.MULTILINE
)

# Extract marker number from annotation line (e.g., "¹ Subs. by..." or "[1] Subs...")
FOOTNOTE_ANNOTATION_MARKER_PATTERN = re.compile(
    rf"^\s*(?:([{UNICODE_SUPERSCRIPTS}])|\[(\d+)\]|(\d+))\s*[.\s]+",
    re.UNICODE
)

# Marker with surrounding context (for section detection)
MARKER_CONTEXT_PATTERN = re.compile(
    rf".{{0,50}}(?:[{UNICODE_SUPERSCRIPTS}]|\[(\d+)\]).{{0,50}}",
    re.DOTALL
)

# ============================================================================
# Amendment Type Definitions
# ============================================================================

AMENDMENT_TYPES = {
    "subs": "substitution",
    "ins": "insertion",
    "omitted": "deletion",
}

# Pattern mapping for type detection
AMENDMENT_TYPE_PATTERNS: dict[str, list[Pattern]] = {
    "subs": [SUBS_PATTERN, SUBS_SIMPLE_PATTERN, SUBS_IBID_PATTERN],
    "ins": [INS_PATTERN, INS_IBID_PATTERN],
    "omitted": [OMITTED_PATTERN, OMITTED_IBID_PATTERN, CLAUSE_OMITTED_PATTERN],
}


# ============================================================================
# Extracted Amendment Structure
# ============================================================================

class ExtractedAmendment:
    """Represents a single amendment extracted from PDF annotations.

    Attributes:
        amendment_type: Type of amendment (substitution, insertion, deletion)
        act_number: Number of the amending act
        act_year: Year of the amending act
        section_number: Section number in the amending act
        target_section: Target section being amended (from main act)
        target_subsection: Target subsection (if specified)
        target_clause: Target clause (if specified)
        effective_date: Date when amendment takes effect (w.e.f.)
        original_text: Original text being replaced (for substitutions)
        ibid_reference: Whether this references the same act as previous
    """

    def __init__(
        self,
        amendment_type: str,
        act_number: str | None = None,
        act_year: str | None = None,
        section_number: str | None = None,
        target_section: str | None = None,
        target_subsection: str | None = None,
        target_clause: str | None = None,
        effective_date: str | None = None,
        original_text: str | None = None,
        ibid_reference: bool = False,
        footnote_marker: str | None = None,
        linkage_method: str = "",
        linkage_confidence: str = "",
    ):
        self.amendment_type = amendment_type
        self.act_number = act_number
        self.act_year = act_year
        self.section_number = section_number
        self.target_section = target_section
        self.target_subsection = target_subsection
        self.target_clause = target_clause
        self.effective_date = effective_date
        self.original_text = original_text
        self.ibid_reference = ibid_reference
        self.footnote_marker = footnote_marker
        self.linkage_method = linkage_method
        self.linkage_confidence = linkage_confidence

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "amendment_type": self.amendment_type,
            "act_number": self.act_number,
            "act_year": self.act_year,
            "section_number": self.section_number,
            "target_section": self.target_section,
            "target_subsection": self.target_subsection,
            "target_clause": self.target_clause,
            "effective_date": self.effective_date,
            "original_text": self.original_text,
            "ibid_reference": self.ibid_reference,
            "footnote_marker": self.footnote_marker,
            "linkage_method": self.linkage_method,
            "linkage_confidence": self.linkage_confidence,
        }

    @property
    def amendment_act_id(self) -> str:
        """Generate amendment act identifier (e.g., 'Act 38 of 1994')."""
        if self.act_number and self.act_year:
            return f"Act {self.act_number} of {self.act_year}"
        return "Unknown"

    def __repr__(self) -> str:
        return (
            f"ExtractedAmendment(type={self.amendment_type}, "
            f"act={self.amendment_act_id}, target={self.target_section}, "
            f"marker={self.footnote_marker})"
        )


@dataclass
class FootnoteContext:
    """Links a footnote marker to its source context.

    Attributes:
        marker: The marker string (e.g., '¹' or '1')
        marker_num: Numeric value of the marker (1-9)
        surrounding_text: Text surrounding the marker
        suspected_section: Section detected from context
        page_num: Page where marker appears
        line_num: Line number on page
    """
    marker: str
    marker_num: int
    surrounding_text: str
    suspected_section: str | None
    page_num: int
    line_num: int


# ============================================================================
# Helper Functions
# ============================================================================

def get_all_patterns() -> list[tuple[str, Pattern]]:
    """Get all amendment patterns with their type labels.

    Returns:
        List of tuples (pattern_name, compiled_pattern)
    """
    return [
        ("subs", SUBS_PATTERN),
        ("subs_simple", SUBS_SIMPLE_PATTERN),
        ("subs_ibid", SUBS_IBID_PATTERN),
        ("ins", INS_PATTERN),
        ("ins_ibid", INS_IBID_PATTERN),
        ("omitted", OMITTED_PATTERN),
        ("omitted_ibid", OMITTED_IBID_PATTERN),
        ("clause_omitted", CLAUSE_OMITTED_PATTERN),
    ]


def detect_amendment_type(text: str) -> str | None:
    """Detect the type of amendment from text.

    Args:
        text: Amendment annotation text

    Returns:
        Amendment type string or None if no match
    """
    text_lower = text.lower()

    if text_lower.startswith("subs."):
        return "substitution"
    elif text_lower.startswith("ins."):
        return "insertion"
    elif "omitted" in text_lower:
        return "deletion"

    return None
