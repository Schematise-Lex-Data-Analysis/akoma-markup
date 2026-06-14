"""Write converted sections to markup and metadata files."""

import json
from datetime import datetime
from pathlib import Path


def write_markup(sections: list[dict], output_path: str) -> str:
    """Write converted entries to a markup text file.

    Section entries (``kind`` is ``"section"`` or absent) are grouped by
    chapter under ``CHAPTER`` dividers. Trailing-table entries
    (``kind == "trailing_table"``) are appended at the end with no wrapper
    or label — just the bluebell TABLE block.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    section_entries = [
        s for s in sections if s.get("kind") != "trailing_table"
    ]
    trailing_tables = [
        s for s in sections if s.get("kind") == "trailing_table"
    ]

    with open(out, "w", encoding="utf-8") as f:
        current_chapter = None
        for sec in section_entries:
            chapter_id = sec.get("chapter_roman", "NA")
            if chapter_id != current_chapter:
                current_chapter = chapter_id
                f.write(f"\n\nCHAPTER {sec.get('chapter_roman', 'NA')}\n")
                f.write(f"{sec.get('chapter_heading', 'Unknown')}\n")
                f.write("=" * 80 + "\n\n")
            f.write(sec["markup"])
            f.write("\n\n")

        for tbl in trailing_tables:
            f.write(tbl["markup"])
            f.write("\n\n")

    return str(out)


def write_versioned_metadata(
    metadata: dict,
    output_path: str,
    version_type: str = "base",
    version_label: str | None = None,
) -> str:
    """Write versioned metadata JSON for amended legislation.

    Args:
        metadata: Metadata dictionary.
        output_path: Path to the markup file.
        version_type: "base" or "amended".
        version_label: Optional version label.

    Returns:
        The metadata file path.
    """
    meta_path = Path(output_path).with_suffix(".meta.json")
    full_metadata = {
        "conversion_date": datetime.now().isoformat(),
        "version_type": version_type,
        **metadata
    }
    if version_label:
        full_metadata["version_label"] = version_label

    with open(meta_path, "w") as f:
        json.dump(full_metadata, f, indent=2)

    return str(meta_path)


def write_metadata(
    sections: list[dict],
    errors: list[dict],
    output_path: str,
    document_name: str | None = None,
    act_number: str | None = None,
    replaces: str | None = None,
) -> str:
    """Write conversion metadata JSON alongside the markup file.

    Args:
        sections: Successfully converted sections.
        errors: Sections that failed conversion.
        output_path: Path to the markup file (metadata is written next to it).
        document_name: Name of the document.
        act_number: Act number.
        replaces: Previous act this document replaces.

    Returns:
        The metadata file path.
    """
    meta_path = Path(output_path).with_suffix(".meta.json")

    metadata = {
        "conversion_date": datetime.now().isoformat(),
        "sections_converted": len(sections),
        "chapters": len({sec.get("chapter_roman", "NA") for sec in sections}),
        "errors": len(errors),
    }

    if document_name:
        metadata["document"] = document_name
    if act_number:
        metadata["act_number"] = act_number
    if replaces:
        metadata["replaces"] = replaces

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return str(meta_path)


def generate_versioned_filename(
    act_name: str,
    version_type: str,
    version_label: str | None = None,
    timestamp: bool = True,
) -> str:
    """Generate a versioned filename for legislation markup.

    Args:
        act_name: Name of the act.
        version_type: "base" or "amended".
        version_label: Optional version label (e.g., "2008").
        timestamp: Whether to include timestamp.

    Returns:
        Generated filename.
    """
    # Clean act name
    safe_act = act_name.lower().replace(" ", "_")
    safe_act = "".join(c for c in safe_act if c.isalnum() or c in "_")

    parts = [safe_act, version_type]
    if version_label:
        parts.append(version_label)
    if timestamp:
        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))

    return f"{'_'.join(parts)}.txt"
