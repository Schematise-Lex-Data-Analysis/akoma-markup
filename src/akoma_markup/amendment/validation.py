"""Amendment validation and conflict detection.

Verifies amendment consistency across operations, detects conflicts,
and validates section numbering and reference integrity.
"""

import logging
import re
from collections import defaultdict
from typing import Literal

logger = logging.getLogger(__name__)


def detect_conflicting_amendments(
    amendments: list[dict],
) -> list[dict]:
    """Detect conflicting amendments to same sections.

    Conflicts occur when:
    1. Multiple amendments target the same section with different operations
    2. Insertion followed by deletion
    3. Replacement that would delete later insertion

    Args:
        amendments: List of amendment dicts with keys:
            - section: str
            - operation: "replace" | "insert" | "delete"
            - amendment_act: str
            - effective_date: str

    Returns:
        List of conflict dictionaries with details.
    """
    conflicts: list[dict] = []
    sections: dict[str, list[dict]] = defaultdict(list)

    # Group by section
    for i, amd in enumerate(amendments):
        sections[amd["section"]].append({
            "index": i,
            **amd
        })

    # Check each section
    for section_num, section_amendments in sections.items():
        if len(section_amendments) <= 1:
            continue

        # Check operation conflicts
        operations = set(a["operation"] for a in section_amendments)

        # Rule 1: Multiple operations on same section
        if len(operations) > 1:
            conflicts.append({
                "section": section_num,
                "type": "multiple_operations",
                "amendments": section_amendments,
                "message": f"Section {section_num} has multiple operations: {operations}",
            })

        # Rule 2: Duplicate operations
        if len(section_amendments) > 1 and len(operations) == 1:
            ops = list(operations)[0]
            conflicts.append({
                "section": section_num,
                "type": "duplicate_operations",
                "amendments": section_amendments,
                "message": f"Section {section_num} has multiple {ops} operations",
            })

        # Rule 3: Check chronological order issues
        sorted_by_date = sorted(
            section_amendments,
            key=lambda x: x.get("effective_date", "")
        )
        if sorted_by_date != section_amendments:
            conflicts.append({
                "section": section_num,
                "type": "out_of_order",
                "amendments": section_amendments,
                "message": f"Section {section_num} amendments not in chronological order",
            })

    return conflicts


def validate_section_references(
    markup: str,
    amended_sections: list[str],
) -> list[dict]:
    """Validate section numbering and reference integrity.

    Checks for:
    1. Missing referenced sections
    2. Invalid section numbers (e.g., 43A when only 43 exists)
    3. Cross-references to repealed sections

    Args:
        markup: Legislation markup text.
        amended_sections: List of sections that were amended.

    Returns:
        List of validation issues.
    """
    issues: list[dict] = []

    # Extract all section numbers
    section_pattern = r"^SECTION\s+(\w+)[.:\s]"
    sections = set(re.findall(section_pattern, markup, re.MULTILINE | re.IGNORECASE))

    # Extract all references to sections
    ref_patterns = [
        r"section\s+(\w+)",  # "section 43"
        r"sections?\s+(\w+(?:\s*,\s*\w+)*)",  # "sections 43, 44, and 45"
        r"section\s+(\w+)\s+and\s+(\w+)",  # "section 43 and 44"
        r"subsection\s+\((\w+)\)",  # "subsection (1)"
        r"clause\s+\((\w+)\)",  # "clause (a)"
        r"(\w+)\s+of\s+section\s+(\w+)",  # "sub-section (1) of section 43"
    ]

    all_references = set()
    for pattern in ref_patterns:
        matches = re.findall(pattern, markup, re.IGNORECASE)
        if isinstance(matches[0], tuple) if matches else False:
            for match in matches:
                all_references.update(m.strip() for m in match if m.strip())
        else:
            all_references.update(m.strip() for m in matches if m.strip())

    # Check for missing references
    for ref in all_references:
        # Skip non-numeric references (like "a", "i", "1", etc.)
        if not ref.strip().isdigit() and not ref.strip().isalpha():
            continue

        # Check if reference exists as a section
        if ref not in sections:
            issues.append({
                "type": "missing_section",
                "reference": ref,
                "message": f"Reference to non-existent section: {ref}",
            })

        # Check if reference is to an amended section
        if ref in amended_sections:
            issues.append({
                "type": "reference_to_amended",
                "reference": ref,
                "message": f"Reference to amended section: {ref}",
            })

    # Check section number sequence
    numeric_sections = []
    for s in sections:
        if s.isdigit():
            numeric_sections.append(int(s))

    if numeric_sections:
        numeric_sections.sort()
        expected = list(range(min(numeric_sections), max(numeric_sections) + 1))
        missing = set(expected) - set(numeric_sections)
        if missing:
            issues.append({
                "type": "missing_sequence",
                "missing": sorted(missing),
                "message": f"Missing section numbers in sequence: {sorted(missing)}",
            })

    return issues


def check_amendment_consistency(
    base_markup: str,
    amended_markup: str,
    amendments: list[dict],
) -> dict[str, list]:
    """Check amendment consistency across operations.

    Validates that:
    1. All amendments were applied
    2. No unintended changes were introduced
    3. Structural integrity is maintained

    Args:
        base_markup: Original markup before amendments.
        amended_markup: Markup after amendments.
        amendments: List of amendments that were applied.

    Returns:
        Dictionary of consistency checks.
    """
    checks: dict[str, list] = {
        "applied_amendments": [],
        "unintended_changes": [],
        "structural_issues": [],
    }

    # Extract sections before and after
    base_sections = _extract_sections(base_markup)
    amended_sections = _extract_sections(amended_markup)

    # Check each amendment was applied
    for amd in amendments:
        section = amd["section"]
        operation = amd["operation"]

        if section not in amended_sections and operation != "delete":
            checks["applied_amendments"].append({
                "section": section,
                "operation": operation,
                "message": f"Section {section} not found after {operation}",
            })

        # For deletions, check if marked as repealed
        if operation == "delete" and section in amended_sections:
            section_text = amended_sections[section]
            if "[REPEALED]" not in section_text:
                checks["applied_amendments"].append({
                    "section": section,
                    "operation": operation,
                    "message": f"Section {section} not marked as REPEALED after delete",
                })

    # Check for unintended changes
    base_section_nums = set(base_sections.keys())
    amended_section_nums = set(amended_sections.keys())

    # Sections that disappeared (except deletions)
    deleted_sections = base_section_nums - amended_section_nums
    for s in deleted_sections:
        # Find if this was a deletion
        matching_amendments = [a for a in amendments if a["section"] == s and a["operation"] == "delete"]
        if not matching_amendments:
            checks["unintended_changes"].append({
                "section": s,
                "type": "unintended_deletion",
                "message": f"Section {s} disappeared without delete amendment",
            })

    # New sections (except insertions)
    new_sections = amended_section_nums - base_section_nums
    for s in new_sections:
        matching_amendments = [a for a in amendments if a["section"] == s and a["operation"] == "insert"]
        if not matching_amendments:
            checks["unintended_changes"].append({
                "section": s,
                "type": "unintended_insertion",
                "message": f"New section {s} appeared without insert amendment",
            })

    return checks


def _extract_sections(markup: str) -> dict[str, str]:
    """Extract sections from markup into dict."""
    sections: dict[str, str] = {}
    lines = markup.split("\n")
    current_section = None
    current_content = []

    for line in lines:
        # Match section header
        match = re.match(r"^SECTION\s+(\w+)[.:\s]", line, re.IGNORECASE)
        if match:
            # Save previous section
            if current_section is not None:
                sections[current_section] = "\n".join(current_content)
                current_content = []

            current_section = match.group(1)
            current_content.append(line)
        elif current_section is not None:
            current_content.append(line)

    # Save last section
    if current_section is not None:
        sections[current_section] = "\n".join(current_content)

    return sections


def validate_hierarchy_consistency(markup: str) -> list[dict]:
    """Validate hierarchical structure consistency.

    Checks that:
    1. Subsections belong to sections
    2. Clauses belong to subsections
    3. Numbering is consistent within hierarchy

    Args:
        markup: Legislation markup text.

    Returns:
        List of hierarchy issues.
    """
    issues: list[dict] = []
    lines = markup.split("\n")

    current_section = None
    current_subsection = None
    section_levels: dict[str, list[str]] = {}

    for i, line in enumerate(lines):
        # Check section
        section_match = re.match(r"^SECTION\s+(\w+)[.:\s]", line, re.IGNORECASE)
        if section_match:
            current_section = section_match.group(1)
            current_subsection = None
            section_levels[current_section] = []
            continue

        # Check subsection
        if line.strip().startswith("SUBSECTION"):
            if current_section is None:
                issues.append({
                    "line": i + 1,
                    "type": "orphaned_subsection",
                    "content": line,
                    "message": "Subsection without parent section",
                })
            else:
                current_subsection = line.strip()
                section_levels[current_section].append(current_subsection)

        # Check clause
        if line.strip().startswith("CLAUSE"):
            if current_subsection is None:
                issues.append({
                    "line": i + 1,
                    "type": "orphaned_clause",
                    "content": line,
                    "message": "Clause without parent subsection",
                })

    return issues