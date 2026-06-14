"""Version diff generation for legislation.

Creates human-readable diff between versions, highlighting
section additions, deletions, and modifications.
"""

import difflib
import logging
import re

logger = logging.getLogger(__name__)


def generate_text_diff(
    markup1: str,
    markup2: str,
    context_lines: int = 3,
) -> str:
    """Generate human-readable diff between two markup texts.

    Uses Python's difflib to create unified diff format.

    Args:
        markup1: First markup (original).
        markup2: Second markup (amended).
        context_lines: Number of context lines around changes.

    Returns:
        Unified diff string.
    """
    lines1 = markup1.splitlines(keepends=True)
    lines2 = markup2.splitlines(keepends=True)

    diff = difflib.unified_diff(
        lines1, lines2,
        fromfile="original",
        tofile="amended",
        n=context_lines,
        lineterm="",
    )

    return "".join(diff)


def generate_section_diff(
    markup1: str,
    markup2: str,
) -> dict:
    """Generate structured diff by section.

    Identifies:
    1. Added sections
    2. Deleted sections
    3. Modified sections
    4. Unchanged sections

    Args:
        markup1: First markup (original).
        markup2: Second markup (amended).

    Returns:
        Dictionary with section-level diff information.
    """
    sections1 = _extract_section_map(markup1)
    sections2 = _extract_section_map(markup2)

    result: dict = {
        "added": [],
        "deleted": [],
        "modified": [],
        "unchanged": [],
    }

    all_sections = set(sections1.keys()) | set(sections2.keys())

    for section_num in sorted(all_sections, key=_section_sort_key):
        in1 = section_num in sections1
        in2 = section_num in sections2

        if in1 and not in2:
            # Deleted section
            result["deleted"].append({
                "section": section_num,
                "content": sections1[section_num],
            })

        elif not in1 and in2:
            # Added section
            result["added"].append({
                "section": section_num,
                "content": sections2[section_num],
            })

        elif in1 and in2:
            content1 = sections1[section_num]
            content2 = sections2[section_num]

            if content1 == content2:
                # Unchanged
                result["unchanged"].append({
                    "section": section_num,
                    "content": content1,
                })
            else:
                # Modified
                result["modified"].append({
                    "section": section_num,
                    "original": content1,
                    "amended": content2,
                    "diff": _generate_section_text_diff(content1, content2),
                })

    return result


def generate_summary_report(
    section_diff: dict,
    amendments: list[dict] | None = None,
) -> str:
    """Generate human-readable summary report of changes.

    Args:
        section_diff: Result from generate_section_diff.
        amendments: Optional list of amendments for context.

    Returns:
        Markdown-formatted summary report.
    """
    report = ["# Amendment Summary Report\n"]

    # Summary statistics
    stats = (
        f"- Added sections: {len(section_diff['added'])}\n"
        f"- Deleted sections: {len(section_diff['deleted'])}\n"
        f"- Modified sections: {len(section_diff['modified'])}\n"
        f"- Unchanged sections: {len(section_diff['unchanged'])}\n"
    )
    report.append("## Summary Statistics\n")
    report.append(stats)

    # Added sections
    if section_diff["added"]:
        report.append("\n## Added Sections\n")
        for added in section_diff["added"]:
            heading = _extract_section_heading(added["content"])
            report.append(f"- Section {added['section']}: {heading}")

    # Deleted sections
    if section_diff["deleted"]:
        report.append("\n## Deleted Sections\n")
        for deleted in section_diff["deleted"]:
            heading = _extract_section_heading(deleted["content"])
            report.append(f"- Section {deleted['section']}: {heading}")

    # Modified sections
    if section_diff["modified"]:
        report.append("\n## Modified Sections\n")
        for modified in section_diff["modified"]:
            heading1 = _extract_section_heading(modified["original"])
            heading2 = _extract_section_heading(modified["amended"])
            report.append(f"- Section {modified['section']}:")
            if heading1 != heading2:
                report.append(f"  Heading: '{heading1}' → '{heading2}'")
            report.append(f"  [Detailed diff available]")

    # Amendment context
    if amendments:
        report.append("\n## Amendment Context\n")
        for amd in amendments:
            report.append(f"- **{amd['amendment_act']}**")
            report.append(f"  - Effective: {amd.get('effective_date', 'N/A')}")
            report.append(f"  - Section(s): {', '.join(amd.get('sections', []))}")
            if "operation" in amd:
                report.append(f"  - Operation: {amd['operation']}")

    # Detailed changes (first few)
    if section_diff["modified"]:
        report.append("\n## Example Changes\n")
        for i, modified in enumerate(section_diff["modified"][:3]):
            report.append(f"### Section {modified['section']}\n")
            report.append("```diff")
            report.append(modified["diff"][:500] + "..." if len(modified["diff"]) > 500 else modified["diff"])
            report.append("```")
            if i < 2:
                report.append("---")

    return "\n".join(report)


def _extract_section_map(markup: str) -> dict[str, str]:
    """Extract sections from markup into dict."""
    sections: dict[str, str] = {}
    lines = markup.split("\n")
    current_section: str | None = None
    current_content: list[str] = []

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


def _section_sort_key(section_num: str) -> tuple:
    """Sort section numbers with alphanumeric support."""
    # Split into numeric and alphabetic parts
    match = re.match(r"^(\d+)([A-Za-z]*)$", section_num)
    if match:
        num = int(match.group(1))
        suffix = match.group(2)
        return (num, suffix)
    # Fallback for non-standard section numbers
    return (0, section_num)


def _generate_section_text_diff(content1: str, content2: str) -> str:
    """Generate diff for a single section."""
    lines1 = content1.splitlines(keepends=True)
    lines2 = content2.splitlines(keepends=True)

    diff = difflib.unified_diff(
        lines1, lines2,
        n=2,
        lineterm="",
    )

    return "".join(diff)


def _extract_section_heading(section_content: str) -> str:
    """Extract heading from section content."""
    lines = section_content.split("\n")
    if not lines:
        return ""

    # First line is "SECTION N. - Heading"
    first_line = lines[0]
    # Extract heading after the dash
    if " - " in first_line:
        return first_line.split(" - ", 1)[1].strip()
    return first_line


def compare_multiple_versions(
    versions: list[tuple[str, str]],  # List of (version_name, markup)
) -> dict:
    """Compare multiple versions to create evolutionary diff.

    Args:
        versions: List of (version_name, markup) pairs in chronological order.

    Returns:
        Dictionary with evolutionary diff.
    """
    if len(versions) < 2:
        return {"error": "Need at least 2 versions to compare"}

    result: dict = {
        "timeline": [],
        "cumulative_changes": {},
    }

    # Compare each version to its predecessor
    for i in range(1, len(versions)):
        prev_name, prev_markup = versions[i - 1]
        curr_name, curr_markup = versions[i]

        section_diff = generate_section_diff(prev_markup, curr_markup)

        result["timeline"].append({
            "from_version": prev_name,
            "to_version": curr_name,
            "changes": section_diff,
        })

    # Calculate cumulative changes from first to last
    if len(versions) > 1:
        first_name, first_markup = versions[0]
        last_name, last_markup = versions[-1]
        result["cumulative_changes"] = generate_section_diff(
            first_markup, last_markup
        )

    return result