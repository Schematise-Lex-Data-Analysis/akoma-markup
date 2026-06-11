"""Complete Gazette conversion pipeline.

Integrates AI-powered section extraction, table rescue, and markup conversion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from ..util.llm.factory import build_llm
from ..util.llm.vision import VisionClient
from ..parsing.tables.rescue import rescue_tables, TableMode
from .ai_extractor import extract_sections_with_ai, save_extraction_tsv

logger = logging.getLogger(__name__)

TableModeValue = Literal["declared", "auto", "full"]


def _run_table_rescue(
    pdf_path: Path,
    tsv_df: pd.DataFrame,
    output_path: Path,
    table_mode: TableModeValue | None,
    vision_client: VisionClient,
    table_pages: list[int] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run table rescue and enhance section content with TABLE_REGION sentinels.

    Args:
        pdf_path: Source PDF path
        tsv_df: DataFrame with extracted sections
        output_path: Output path for cache anchoring
        table_mode: Table detection mode
        vision_client: VisionClient for table extraction
        table_pages: Specific pages with tables (for 'declared' mode)

    Returns:
        Tuple of (enhanced DataFrame, table regions list)
    """
    if table_mode is None:
        return tsv_df, []

    # Get per-page text by aggregating section content
    max_page = tsv_df["page"].max() if not tsv_df.empty else 0
    per_page_text: list[str] = []

    for page_num in range(1, max_page + 1):
        page_sections = tsv_df[tsv_df["page"] == page_num]
        if page_sections.empty:
            per_page_text.append("")
        else:
            content = "\n\n".join(
                f"{row['section_num']}. {row['heading']}\n{row['content']}"
                for _, row in page_sections.iterrows()
            )
            per_page_text.append(content)

    try:
        enhanced_text, table_regions = rescue_tables(
            pdf_path=pdf_path,
            per_page_text=per_page_text,
            output_path=str(output_path),
            mode=table_mode,
            azure_vision_key=vision_client.api_key,
            azure_vision_endpoint=vision_client.endpoint,
            azure_vision_model=vision_client.deployment,
            azure_vision_api_style=vision_client.api_mode.value,
            table_pages=table_pages,
            azure_vision_max_tokens=vision_client.extraction_max_tokens,
        )

        # Update DataFrame content with sentinels
        for page_num, text in enumerate(enhanced_text, 1):
            # Check for TABLE_REGION sentinels in this page
            import re
            sentinels = re.findall(r"<<TABLE_REGION:(\d+)>>", text)

            if sentinels:
                # Find the section(s) that should contain the table
                page_mask = tsv_df["page"] == page_num
                for idx in tsv_df[page_mask].index:
                    for sentinel in sentinels:
                        # Append sentinel to appropriate section content
                        current_content = tsv_df.at[idx, "content"]
                        tsv_df.at[idx, "content"] = (
                            f"{current_content}\n\n<<TABLE_REGION:{sentinel}>>"
                        )

        logger.info("Table rescue complete: %d regions found", len(table_regions))
        return tsv_df, table_regions

    except Exception as exc:
        logger.warning("Table rescue failed: %s", exc)
        return tsv_df, []


def convert_gazette(
    pdf_path: Path,
    output_path: Path,
    llm_config: dict,
    vision_client: VisionClient,
    table_mode: TableModeValue | None = None,
    table_pages: list[int] | None = None,
    max_workers: int = 4,
    dpi: int = 200,
    skip_extraction: bool = False,
) -> dict:
    """Convert a Gazette PDF to Akoma Ntoso markup.

    This is the main entry point for the complete Gazette conversion pipeline.

    Args:
        pdf_path: Path to the Gazette PDF
        output_path: Path for the output markup file
        llm_config: Configuration for the text conversion LLM
        vision_client: VisionClient for AI extraction and table rescue
        table_mode: Table detection mode ("declared", "auto", "full", or None)
        table_pages: Specific pages with tables (for 'declared' mode)
        max_workers: Number of parallel workers for page analysis
        dpi: DPI for rendering PDF pages to images
        skip_extraction: If True, skip extraction and use existing TSV

    Returns:
        Dictionary with conversion results and metadata
    """
    logger.info("══ Gazette Conversion Pipeline ══")
    logger.info("PDF: %s", pdf_path)
    logger.info("Output: %s", output_path)

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = output_dir / f"{pdf_path.stem}_sections.tsv"

    # Phase 1: AI-Powered Section Extraction
    if skip_extraction and tsv_path.exists():
        logger.info("Loading existing TSV: %s", tsv_path)
        tsv_df = pd.read_csv(tsv_path, sep="\t")
    else:
        logger.info("══ Phase 1: AI-Powered Section Extraction ══")
        tsv_df = extract_sections_with_ai(
            pdf_path=pdf_path,
            vision_client=vision_client,
            output_dir=output_dir,
            max_workers=max_workers,
            dpi=dpi,
        )

        # Save to TSV
        save_extraction_tsv(tsv_df, tsv_path)

    # Phase 2: Multimodal Table Integration (if enabled)
    if table_mode:
        logger.info("══ Phase 2: Table Rescue ══")
        tsv_df, table_regions = _run_table_rescue(
            pdf_path=pdf_path,
            tsv_df=tsv_df,
            output_path=output_path,
            table_mode=table_mode,
            vision_client=vision_client,
            table_pages=table_pages,
        )
    else:
        table_regions = []

    # Phase 3: AI-Powered Markup Conversion
    logger.info("══ Phase 3: Markup Conversion ══")
    markup = _convert_sections_to_markup(
        tsv_df=tsv_df,
        table_regions=table_regions,
        llm_config=llm_config,
        document_name=pdf_path.stem,
        output_path=output_path,
    )

    # Write output
    output_path.write_text(markup, encoding="utf-8")
    logger.info("Wrote markup to: %s", output_path)

    # Create metadata
    metadata = {
        "pdf_path": str(pdf_path),
        "output_path": str(output_path),
        "tsv_path": str(tsv_path),
        "sections_extracted": len(tsv_df),
        "table_regions": len(table_regions),
        "table_mode": table_mode,
    }

    return {
        "markup": markup,
        "metadata": metadata,
        "tsv_df": tsv_df,
        "table_regions": table_regions,
    }


def _convert_sections_to_markup(
    tsv_df: pd.DataFrame,
    table_regions: list[dict],
    llm_config: dict,
    document_name: str,
    output_path: Path,
) -> str:
    """Convert TSV sections to Akoma Ntoso markup using per-section LLM calls.

    This is Phase 3: AI-powered markup conversion.

    Args:
        tsv_df: DataFrame with extracted sections
        table_regions: List of table region dicts with markdown
        llm_config: Configuration for text conversion LLM
        document_name: Name of the document
        output_path: Path for the output file (used for checkpoint location)

    Returns:
        Akoma Ntoso markup string
    """
    from ..util.llm.factory import build_llm
    from .conversion import (
        build_gazette_chain,
        process_gazette_sections,
        assemble_gazette_markup,
    )

    # Build LLM and chain (mirroring original pattern)
    llm = build_llm(dict(llm_config))
    chain = build_gazette_chain(llm, document_name)

    # Prepare sections from DataFrame (matching conversion.py format)
    sections = []
    for _, row in tsv_df.iterrows():
        sections.append({
            "num": row["section_num"],
            "heading": row["heading"],
            "content": row["content"],
            "hierarchy_level": row.get("hierarchy_level", 1),
            "parent_section": row.get("parent_section", ""),
            "page": row["page"],
        })

    # Setup checkpoint path
    checkpoint_dir = output_path.parent / ".akoma_cache"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{output_path.stem}_markup_checkpoint.json"

    # Process all sections (mirrors process_all_sections pattern)
    results, errors = process_gazette_sections(
        chain=chain,
        sections=sections,
        checkpoint_path=checkpoint_path,
    )

    # Log errors
    if errors:
        logger.warning("%d sections failed conversion", len(errors))
        for err in errors:
            logger.warning("  Section %s: %s", err["num"], err["error"][:60])

    # Assemble document from results
    markup = assemble_gazette_markup(results, document_name, table_regions)

    return markup


class GazetteConverter:
    """Stateful converter for multiple gazette documents.

    Maintains configuration and vision client for batch processing.
    """

    def __init__(
        self,
        llm_config: dict,
        vision_client: VisionClient,
        output_dir: Path,
        table_mode: TableModeValue | None = None,
        max_workers: int = 4,
        dpi: int = 200,
    ):
        """Initialize the converter with configuration.

        Args:
            llm_config: Configuration for text conversion LLM
            vision_client: VisionClient for AI extraction
            output_dir: Base directory for outputs
            table_mode: Table detection mode
            max_workers: Number of parallel workers
            dpi: DPI for page rendering
        """
        self.llm_config = llm_config
        self.vision_client = vision_client
        self.output_dir = Path(output_dir)
        self.table_mode = table_mode
        self.max_workers = max_workers
        self.dpi = dpi

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build text conversion LLM
        self._llm = build_llm(dict(llm_config))

    def convert(self, pdf_path: Path, document_name: str | None = None) -> dict:
        """Convert a single gazette document.

        Args:
            pdf_path: Path to the Gazette PDF
            document_name: Optional document name (defaults to PDF stem)

        Returns:
            Conversion results dictionary
        """
        name = document_name or pdf_path.stem
        output_path = self.output_dir / f"{name}.txt"

        return convert_gazette(
            pdf_path=pdf_path,
            output_path=output_path,
            llm_config=self.llm_config,
            vision_client=self.vision_client,
            table_mode=self.table_mode,
            max_workers=self.max_workers,
            dpi=self.dpi,
        )

    def convert_batch(self, pdf_paths: list[Path]) -> list[dict]:
        """Convert multiple gazette documents.

        Args:
            pdf_paths: List of PDF paths to convert

        Returns:
            List of conversion results
        """
        results = []
        for pdf_path in pdf_paths:
            try:
                result = self.convert(pdf_path)
                results.append(result)
            except Exception as exc:
                logger.error("Failed to convert %s: %s", pdf_path, exc)
                results.append({
                    "error": str(exc),
                    "pdf_path": str(pdf_path),
                })
        return results
