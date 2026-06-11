"""AI-powered section extraction for Gazette notifications.

Uses multimodal vision LLM to intelligently identify document structure,
extract sections with hierarchies, and filter multilingual content.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from PIL import Image

from ..util.llm.vision import VisionClient
from ..util.pdf.images import render_pages, page_count
from .prompts import GAZETTE_SECTION_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSection:
    """Represents a single section extracted from a Gazette page."""

    page: int
    section_num: str
    heading: str
    content: str
    chapter: str = ""
    start_pos: int = 0
    end_pos: int = 0
    language: str = "en"
    confidence: float = 0.0
    hierarchy_level: int = 1
    parent_section: str = ""
    bbox: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for TSV export."""
        return {
            "page": self.page,
            "section_num": self.section_num,
            "heading": self.heading,
            "content": self.content,
            "chapter": self.chapter,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "language": self.language,
            "confidence": self.confidence,
            "hierarchy_level": self.hierarchy_level,
            "parent_section": self.parent_section,
        }


class ExtractionCache:
    """Cache for AI extraction results to enable resumable processing."""

    def __init__(self, pdf_path: Path, output_dir: Path):
        self.pdf_path = pdf_path
        self.cache_dir = output_dir / ".akoma_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"{pdf_path.stem}_gazette_extraction.json"
        self._data: dict[str, Any] = self._load()
        self._lock = Lock()

    def _load(self) -> dict[str, Any]:
        """Load cache from disk if valid."""
        if not self.cache_file.exists():
            return {"pages": {}, "pdf_mtime": None}

        try:
            with open(self.cache_file) as f:
                data = json.load(f)

            # Validate cache against PDF modification time
            current_mtime = self.pdf_path.stat().st_mtime
            if data.get("pdf_mtime") != current_mtime:
                logger.info("Cache invalidated: PDF mtime changed")
                return {"pages": {}, "pdf_mtime": current_mtime}

            return data
        except Exception as exc:
            logger.warning("Failed to load cache: %s", exc)
            return {"pages": {}, "pdf_mtime": None}

    def get_page(self, page_num: int) -> list[dict] | None:
        """Get cached sections for a page."""
        return self._data.get("pages", {}).get(str(page_num))

    def set_page(self, page_num: int, sections: list[dict]) -> None:
        """Cache sections for a page and save to disk."""
        with self._lock:
            self._data["pages"][str(page_num)] = sections
            self._data["pdf_mtime"] = self.pdf_path.stat().st_mtime
            self._save()

    def _save(self) -> None:
        """Save cache to disk atomically."""
        tmp = self.cache_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.cache_file)


def _clean_json_response(text: str) -> str:
    """Clean LLM response by removing markdown code fences."""
    # Remove markdown code blocks
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    # Remove leading/trailing whitespace
    return text.strip()


def _parse_section_analysis(response: str, page_num: int) -> list[dict]:
    """Parse JSON response from vision LLM into section data."""
    try:
        cleaned = _clean_json_response(response)
        data = json.loads(cleaned)

        # Handle both single page and batch formats
        if isinstance(data, list):
            # Find the page in the array
            for item in data:
                if item.get("page_number") == page_num:
                    return item.get("sections", [])
            return []

        if isinstance(data, dict):
            return data.get("sections", [])

        return []
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse JSON response for page %d: %s", page_num, exc)
        # Try to extract JSON from the response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get("sections", [])
            except Exception:
                pass
        return []


def _analyze_single_page(
    vision_client: VisionClient,
    page_image: Image.Image,
    page_num: int,
    previous_context: str = "",
) -> list[dict]:
    """Analyze a single page with vision LLM.

    Args:
        vision_client: VisionClient for making LLM calls
        page_image: PIL Image of the page
        page_num: 1-indexed page number
        previous_context: Context from previous pages for continuity

    Returns:
        List of section dictionaries
    """
    # Build prompt with context if available
    if previous_context:
        prompt = GAZETTE_SECTION_ANALYSIS_PROMPT.replace(
            "Analyze this page from an Indian Gazette notification.",
            f"Analyze this page from an Indian Gazette notification.\n\n"
            f"CONTINUATION CONTEXT:\n"
            f"The previous page ended with:\n{previous_context[-500:]}\n\n"
        )
    else:
        prompt = GAZETTE_SECTION_ANALYSIS_PROMPT

    try:
        response = vision_client.ask(
            image=page_image,
            prompt=prompt,
            detail="high",
            max_tokens=8192,
        )

        sections = _parse_section_analysis(response, page_num)

        # Add page number to each section
        for section in sections:
            section["page"] = page_num

        return sections

    except Exception as exc:
        logger.error("Error analyzing page %d: %s", page_num, exc)
        return []


def analyze_page_parallel(
    vision_client: VisionClient,
    page_images: dict[int, Image.Image],
    cache: ExtractionCache,
    max_workers: int = 4,
) -> dict[int, list[dict]]:
    """Analyze multiple pages in parallel using ThreadPoolExecutor.

    Args:
        vision_client: VisionClient for making LLM calls
        page_images: Dict mapping page numbers to PIL Images
        cache: ExtractionCache for caching results
        max_workers: Number of parallel workers

    Returns:
        Dict mapping page numbers to lists of section dictionaries
    """
    results: dict[int, list[dict]] = {}

    # Check cache first
    pages_to_process = []
    for page_num in sorted(page_images.keys()):
        cached = cache.get_page(page_num)
        if cached is not None:
            logger.debug("Cache hit for page %d", page_num)
            results[page_num] = cached
        else:
            pages_to_process.append(page_num)

    if not pages_to_process:
        return results

    logger.info("Analyzing %d page(s) with vision LLM", len(pages_to_process))

    lock = Lock()

    def _process_page(page_num: int) -> tuple[int, list[dict]]:
        """Process a single page and cache result."""
        previous_context = ""
        if page_num > 1 and page_num - 1 in results:
            prev_content = results[page_num - 1]
            if prev_content:
                last_section = prev_content[-1]
                previous_context = last_section.get("content", "")[-500:]

        sections = _analyze_single_page(
            vision_client=vision_client,
            page_image=page_images[page_num],
            page_num=page_num,
            previous_context=previous_context,
        )

        # Cache the result
        cache.set_page(page_num, sections)

        with lock:
            results[page_num] = sections

        return page_num, sections

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_page, pnum): pnum
            for pnum in pages_to_process
        }

        for future in as_completed(futures):
            page_num = futures[future]
            try:
                future.result()
                logger.info("Completed analysis of page %d", page_num)
            except Exception as exc:
                logger.error("Failed to analyze page %d: %s", page_num, exc)

    return results


def extract_sections_with_ai(
    pdf_path: Path,
    vision_client: VisionClient,
    output_dir: Path,
    max_workers: int = 4,
    dpi: int = 200,
) -> pd.DataFrame:
    """Extract sections from a Gazette PDF using AI-powered analysis.

    This is the main entry point for Phase 1 of the Gazette conversion pipeline.
    Uses multimodal vision LLM to intelligently identify document structure,
    sections, hierarchies, and filter non-English content.

    Args:
        pdf_path: Path to the Gazette PDF
        vision_client: VisionClient instance for making LLM calls
        output_dir: Directory for cache and output files
        max_workers: Number of parallel workers for page analysis
        dpi: DPI for rendering PDF pages to images

    Returns:
        DataFrame with extracted sections in TSV-compatible format
    """
    logger.info("══ Starting AI-Powered Section Extraction ══")
    logger.info("PDF: %s", pdf_path)

    total_pages = page_count(pdf_path)
    logger.info("Total pages: %d", total_pages)

    # Initialize cache
    cache = ExtractionCache(pdf_path, output_dir)

    # Check if we have all pages cached
    all_cached = all(cache.get_page(p) is not None for p in range(1, total_pages + 1))
    if all_cached:
        logger.info("All pages found in cache, skipping analysis")
        sections_data = []
        for p in range(1, total_pages + 1):
            sections_data.extend(cache.get_page(p) or [])
    else:
        # Render all pages
        logger.info("Rendering pages at %d DPI", dpi)
        page_nums = list(range(1, total_pages + 1))
        page_images = render_pages(pdf_path, page_nums, dpi=dpi)

        # Analyze pages
        results = analyze_page_parallel(
            vision_client=vision_client,
            page_images=page_images,
            cache=cache,
            max_workers=max_workers,
        )

        # Collect all sections
        sections_data = []
        for page_num in sorted(results.keys()):
            sections_data.extend(results[page_num])

    logger.info("Extracted %d sections total", len(sections_data))

    # Convert to DataFrame
    df = _sections_to_dataframe(sections_data)

    # Filter to English content
    english_df = df[df["language"].isin(["en", "mixed"])].copy()
    logger.info("After language filter: %d sections", len(english_df))

    return english_df


def _sections_to_dataframe(sections: list[dict]) -> pd.DataFrame:
    """Convert list of section dicts to DataFrame.

    Args:
        sections: List of section dictionaries from AI extraction

    Returns:
        DataFrame with standardized columns
    """
    rows = []
    for sec in sections:
        row = {
            "page": sec.get("page", 0),
            "section_num": sec.get("number", ""),
            "heading": sec.get("heading", ""),
            "content": sec.get("content", ""),
            "chapter": sec.get("chapter", ""),
            "start_pos": sec.get("start_pos", 0),
            "end_pos": sec.get("end_pos", 0),
            "language": sec.get("language", "en"),
            "confidence": sec.get("confidence", 0.0),
            "hierarchy_level": sec.get("hierarchy_level", 1),
            "parent_section": sec.get("parent_section", ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            "page", "section_num", "heading", "content", "chapter",
            "start_pos", "end_pos", "language", "confidence",
            "hierarchy_level", "parent_section"
        ])

    return df


def save_extraction_tsv(df: pd.DataFrame, output_path: Path) -> None:
    """Save extracted sections to TSV file.

    Args:
        df: DataFrame with extracted sections
        output_path: Path to write TSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info("Saved extraction TSV to: %s", output_path)
