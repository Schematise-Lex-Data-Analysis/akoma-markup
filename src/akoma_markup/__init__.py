"""akoma-markup: Convert legislative PDFs to Akoma Ntoso markup."""

import re
import sys
from pathlib import Path

from .converter import build_chain, process_all_sections
from .extractor import extract_pdf_pages, extract_pdf_text
from .llm import build_llm
from .parser import (
    extract_chapter_ranges,
    extract_section_content,
    filter_sections_by_chapters,
    parse_toc,
)
from .writer import write_markup, write_metadata, write_ocr_text

# Azure AI services (optional)
try:
    from .table_ocr_ai import (
        AzureOCR,
        AzureMultimodalAnalyzer,
        IndiaCodeAnalyzer,
        extract_indiacode_tables,
        test_azure_connectivity
    )
    AZURE_AI_AVAILABLE = True
except ImportError:
    AZURE_AI_AVAILABLE = False


def convert(
    pdf_path: str,
    llm_config: dict,
    output_path: str | None = None,
    document_name: str | None = None,
    act_number: str | None = None,
    replaces: str | None = None,
    table_mode: str | None = None,
    table_pages: list[int] | None = None,
    azure_api_key: str | None = None,
    azure_ocr_endpoint: str | None = None,
    azure_multimodal_endpoint: str | None = None,
    azure_multimodal_deployment: str | None = None,
    azure_multimodal_api_style: str | None = None,
) -> str:
    """Convert a legislative PDF to Akoma Ntoso markup.

    Args:
        pdf_path: Path to the legislative PDF file.
        llm_config: LLM provider config dict. Must include 'provider' key.
            Example: {"provider": "openai", "model": "gpt-4o", "api_key": "sk-..."}
        output_path: Destination for the markup file.
            Defaults to ``<pdf_stem>_markup.txt`` in the same directory.
        document_name: Name of the document (e.g., "Bharatiya Nagarik Suraksha Sanhita 2023").
            Defaults to PDF filename stem.
        act_number: Act number (e.g., "46 of 2023").
        replaces: Previous act this document replaces (e.g., "Criminal Procedure Code (CrPC) 1973").
        table_mode: Optional table-rescue strategy. One of "declared" or "auto".
            When None (default), only pdfplumber is used and tables in the PDF may be
            garbled in the output. When set, table regions are additionally sent to
            Azure OCR and converted to Laws.Africa TABLE blocks. Requires `azure_api_key`.
            "auto" additionally requires `azure_multimodal_endpoint` and
            `azure_multimodal_deployment` (or the matching env vars) for the
            vision-LLM page classification step.
        table_pages: 1-indexed page list. Required when `table_mode="declared"`.
        azure_api_key: Azure API key. Required when `table_mode` is set.
        azure_ocr_endpoint: Override the Azure Document Intelligence endpoint.
        azure_multimodal_endpoint: Vision-LLM endpoint (auto mode).
        azure_multimodal_deployment: Vision-LLM deployment name (auto mode).

    Returns:
        Path to the generated markup file.
    """
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    if output_path is None:
        output_path = str(pdf.with_name(f"{pdf.stem}_markup.txt"))

    # Set defaults for document metadata
    if document_name is None:
        document_name = pdf.stem

    if table_mode is not None:
        if table_mode not in {"declared", "auto", "full"}:
            raise ValueError(
                f"table_mode must be 'declared', 'auto', or 'full'; "
                f"got {table_mode!r}"
            )
        if not azure_api_key:
            raise ValueError("table_mode requires azure_api_key")
        if table_mode == "declared" and not table_pages:
            raise ValueError("table_mode='declared' requires table_pages")

    llm = build_llm(llm_config)

    # 1. Extract text (per-page so table rescue can swap individual pages)
    print("Extracting text from PDF ...", file=sys.stderr)
    per_page_text = extract_pdf_pages(str(pdf))

    rescued_pages: dict[int, str] = {}
    if table_mode is not None:
        from .tables import rescue_tables
        print(
            f"Rescuing tables via Azure OCR (mode={table_mode!r}) ...",
            file=sys.stderr,
        )
        per_page_text, rescued_pages = rescue_tables(
            pdf_path=pdf,
            per_page_text=per_page_text,
            mode=table_mode,
            azure_api_key=azure_api_key,
            table_pages=table_pages,
            azure_ocr_endpoint=azure_ocr_endpoint,
            azure_multimodal_endpoint=azure_multimodal_endpoint,
            azure_multimodal_deployment=azure_multimodal_deployment,
            azure_multimodal_api_style=azure_multimodal_api_style,
        )

    raw_text = "\n".join(per_page_text)

    raw_text_debug_path = Path(output_path).with_suffix(".raw_text_debug.txt")
    raw_text_debug_path.write_text(raw_text)
    print(f"Raw text (post-rescue) written to {raw_text_debug_path}", file=sys.stderr)

    all_lines = raw_text.splitlines()

    # 2. Parse TOC
    print("Parsing table of contents ...", file=sys.stderr)
    _chapters, section_names, toc_end_line = parse_toc(all_lines)
    chapter_ranges = extract_chapter_ranges(all_lines, section_names, toc_end_line)
    print(
        f"Found {len(chapter_ranges)} chapters, {len(section_names)} sections "
        f"(TOC ends at line {toc_end_line})",
        file=sys.stderr,
    )

    # 3. Extract section content (skip TOC)
    print(f"OCR text (raw): {all_lines[toc_end_line + 1:][:10]}", file=sys.stderr)
    content_text = "\n".join(all_lines[toc_end_line + 1:])
    sections = extract_section_content(content_text, section_names)

    # 4. Map sections to chapters and deduplicate
    sections = filter_sections_by_chapters(sections, chapter_ranges)

    debug_tsv_path = Path(output_path).with_suffix(".sections_debug.tsv")
    with open(debug_tsv_path, 'a') as debug_file:
        import csv
        writer = csv.writer(debug_file, delimiter='\t')
        for sec in sections:
            content = sec.get('content', '[No content]')
            content_preview = content.replace('\n', ' ')
            writer.writerow([sec['num'], sec['heading'], content_preview])
            print(f"Section debug: number={sec['num']}, heading={sec['heading']}", file=sys.stderr)
    seen = set()
    unique = []
    for sec in sections:
        if sec["num"] not in seen:
            seen.add(sec["num"])
            unique.append(sec)
    sections = unique
    print(f"{len(sections)} unique sections ready for conversion", file=sys.stderr)

    # Perform OCR text writing before LLM conversion
    ocr_path = write_ocr_text(content_text, output_path)
    print(f"OCR text written to {ocr_path}", file=sys.stderr)

    # 5. Convert via LLM
    chain = build_chain(llm, document_name=document_name)
    checkpoint_dir = Path(output_path).parent / ".akoma_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_filename = f"{pdf.stem}_conversion_checkpoint.json"
    checkpoint_path = checkpoint_dir / checkpoint_filename
    converted, errors = process_all_sections(
        chain, sections, checkpoint_path=checkpoint_path
    )

    # Merge chapter info
    sec_lookup = {s["num"]: s for s in sections}
    for conv in converted:
        orig = sec_lookup.get(conv["num"], {})
        conv["chapter_roman"] = orig.get("chapter_roman", "NA")
        conv["chapter_heading"] = orig.get("chapter_heading", "Unknown")

    # 6. Write output
    ocr_path = write_ocr_text(content_text, output_path)
    markup_path = write_markup(converted, output_path)
    meta_path = write_metadata(
        converted, errors, output_path,
        document_name=document_name,
        act_number=act_number,
        replaces=replaces
    )
    #print(f"OCR text written to {ocr_path}", file=sys.stderr)

    print(f"Markup written to {markup_path}", file=sys.stderr)
    print(f"Metadata written to {meta_path}", file=sys.stderr)
    if errors:
        print(f"{len(errors)} sections failed conversion", file=sys.stderr)

    return markup_path


# Azure AI analysis functions
if AZURE_AI_AVAILABLE:
    
    def analyze_indiacode_tables(
        pdf_path: str,
        api_key: str,
        output_dir: str | None = None,
        analysis_types: list[str] | None = None
    ) -> dict:
        """
        Analyze IndiaCode legislative tables using Azure AI services.
        
        Args:
            pdf_path: Path to PDF file
            api_key: Azure API key
            output_dir: Directory for output files
            analysis_types: List of analysis types (summary, structure, content, json)
            
        Returns:
            Dictionary with analysis results
        """
        if analysis_types is None:
            analysis_types = ["summary", "structure", "content"]
        
        return extract_indiacode_tables(
            pdf_path=pdf_path,
            api_key=api_key,
            output_dir=output_dir
        )
    
    def extract_with_document_intelligence(
        pdf_path: str,
        api_key: str,
        output_dir: str | None = None
    ) -> str:
        """
        Extract text from PDF using Azure Document Intelligence (OCR).
        
        Args:
            pdf_path: Path to PDF file
            api_key: Azure API key
            output_dir: Directory to save extracted text
            
        Returns:
            Path to extracted text file
        """
        from pathlib import Path
        
        pdf_path_obj = Path(pdf_path)
        if output_dir:
            output_dir_obj = Path(output_dir)
        else:
            output_dir_obj = pdf_path_obj.parent
        
        ocr = AzureOCR(api_key=api_key)
        return str(ocr.extract_and_save(pdf_path_obj, output_dir_obj))
    
    def test_azure_ai_services(api_key: str) -> bool:
        """
        Test connectivity to Azure AI services.
        
        Args:
            api_key: Azure API key
            
        Returns:
            True if services are accessible
        """
        return test_azure_connectivity(api_key)
