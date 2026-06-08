"""PDF text extraction for indiacode 2023 law documents."""

import pdfplumber


def extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Concatenated text from all pages.
    """
    return "\n".join(extract_pdf_pages(pdf_path))


def extract_pdf_pages(pdf_path: str) -> list[str]:
    """Extract per-page text from a PDF file.

    The returned list has one entry per page (1-indexed externally, 0-indexed here);
    empty pages become empty strings so indices line up with page numbers.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of strings, one per page, preserving page order.
    """
    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            pages.append(text or "")
    return pages
