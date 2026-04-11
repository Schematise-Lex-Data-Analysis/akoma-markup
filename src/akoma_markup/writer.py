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
    vma_path = Path(output_path).with_suffix(".vma.txt")

    ocr_path.write_text(text, encoding="utf-8")

    return str(ocr_path)


def write_markup(sections: list[dict], output_path: str) -> str:
    """Write converted sections to a markup text file grouped by chapter.

    Args:
        sections: List of section dicts with 'markup', 'chapter_roman', 'chapter_heading'.
        output_path: Destination file path.

    Returns:
        The resolved output path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        current_chapter = None
        for sec in sections:
            chapter_id = sec.get("chapter_roman", "NA")
            if chapter_id != current_chapter:
                current_chapter = chapter_id
                f.write(f"\n\nCHAPTER {sec.get('chapter_roman', 'NA')}\n")
                f.write(f"{sec.get('chapter_heading', 'Unknown')}\n")
                f.write("=" * 80 + "\n\n")
            f.write(sec["markup"])
            f.write("\n\n")

    return str(out)


def write_metadata(
    sections: list[dict],
    errors: list[dict],
    output_path: str,
) -> str:
    """Write conversion metadata JSON alongside the markup file.

    Args:
        sections: Successfully converted sections.
        errors: Sections that failed conversion.
        output_path: Path to the markup file (metadata is written next to it).

    Returns:
        The metadata file path.
    """
    meta_path = Path(output_path).with_suffix(".meta.json")

    metadata = {
        "document": "Bharatiya Nagarik Suraksha Sanhita 2023",
        "act_number": "46 of 2023",
        "replaces": "Criminal Procedure Code (CrPC) 1973",
        "conversion_date": datetime.now().isoformat(),
        "sections_converted": len(sections),
        "chapters": len({sec.get("chapter_roman", "NA") for sec in sections}),
        "errors": len(errors),
        "ocr_file": str(Path(output_path).with_suffix(".ocr.txt")),
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return str(meta_path)
