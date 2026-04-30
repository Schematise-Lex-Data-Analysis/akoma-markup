"""Write converted sections to markup and metadata files."""

import json
from datetime import datetime
from pathlib import Path


def write_ocr_text(text: str, output_path: str) -> str:
    """Write the raw OCR text to a file alongside the markup output.

    Args:
        text: The full extracted OCR text from the PDF.
        output_path: Path to the markup file (OCR text is written next to it).

    Returns:
        The OCR text file path.
    """
    ocr_path = Path(output_path).with_suffix(".ocr.txt")
    ocr_path.write_text(text, encoding="utf-8")
    return str(ocr_path)


def write_markup(sections: list[dict], output_path: str) -> str:
    """Write converted entries to a markup text file.

    Section entries (``kind`` is ``"section"`` or absent) are grouped by
    chapter under ``CHAPTER`` dividers. Schedule entries (``kind ==
    "schedule"``) are emitted at the end under a single ``SCHEDULES``
    divider with no chapter wrapper — they're top-level structural
    elements, peers of chapters, not children of one.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    section_entries = [s for s in sections if s.get("kind") != "schedule"]
    schedule_entries = [s for s in sections if s.get("kind") == "schedule"]

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

        if schedule_entries:
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("SCHEDULES\n")
            f.write("=" * 80 + "\n\n")
            for sch in schedule_entries:
                pages = sch.get("pages") or []
                if pages:
                    span = (
                        f"page {pages[0]}"
                        if len(pages) == 1
                        else f"pages {pages[0]}-{pages[-1]}"
                    )
                    f.write(f"# Source: {span}\n")
                f.write(sch["markup"])
                f.write("\n\n")

    return str(out)


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
        "ocr_file": str(Path(output_path).with_suffix(".ocr.txt")),
    }

    # Add document metadata if provided
    if document_name:
        metadata["document"] = document_name
    if act_number:
        metadata["act_number"] = act_number
    if replaces:
        metadata["replaces"] = replaces

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return str(meta_path)
