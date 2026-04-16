"""akoma-markup: Convert legislative PDFs to Akoma Ntoso markup."""

import sys
from pathlib import Path

from .converter import build_chain, process_all_sections
from .extractor import extract_pdf_text
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

    llm = build_llm(llm_config)

    # 1. Extract text
    print("Extracting text from PDF ...", file=sys.stderr)
    raw_text = extract_pdf_text(str(pdf))
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

    with open('/home/stns/akoma-markup/akoma-markup/sections_debug.tsv', 'a') as debug_file:
        for sec in sections:
            import csv
            writer = csv.writer(debug_file, delimiter='\t')
            # Truncate and sanitize content for readable debug output
            content = sec.get('content', '[No content]')
            #content_preview = content.replace('\n', ' ')[:200]
            content_preview = content.replace('\n', ' ')
            if len(content) > 200:
                pass
                #content_preview += '...'
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
