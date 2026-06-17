"""Amendment-related functionality for akoma-markup."""

from .apply import (
    apply_amendment,
    apply_multiple_amendments,
    extract_section_from_markup,
    validate_amended_markup,
)
from .conversion import (
    build_gazette_chain,
    convert_gazette_text,
    convert_gazette_page_with_vision,
    convert_gazette_pages_with_vision,
    merge_gazette_pages,
)
from .diff import (
    generate_section_diff,
    generate_summary_report,
    generate_text_diff,
)
from .extract import (
    AmendmentExtractionResult,
    extract_amendments_from_pdf,
    extract_section_context,
    generate_details_tsv,
    generate_registry_csv,
)
from .footnote_linker import (
    build_page_section_map,
    extract_marker_from_annotation,
    extract_markers_with_context,
    link_footnotes_to_amendments,
    validate_section_linkages,
)
from .patterns import (
    ExtractedAmendment,
    FootnoteContext,
)
from .vision_schema import (
    VisionExtractedAmendment,
    AmendmentExtractionResult,
)
from .vision_extractor import (
    VisionAmendmentExtractor,
    ExtractionConfig,
    extract_amendments_with_vision,
)
from .vision_cache import (
    VisionExtractionCache,
    get_default_cache,
    clear_pdf_cache,
    get_cache_stats,
)
from .hybrid_extractor import (
    HybridAmendmentExtractor,
    HybridExtractionConfig,
    HybridExtractionResult,
    extract_amendments_hybrid,
)
from .evaluation import (
    ExtractionEvaluator,
    EvaluationReport,
    evaluate_extraction_methods,
)
from .registry import (
    AmendmentDetail,
    AmendmentRecord,
    get_amendment_timeline,
    group_amendments_by_act,
    parse_csv_registry,
    parse_tsv_details,
    validate_amendment_records,
)
from .validation import (
    check_amendment_consistency,
    detect_conflicting_amendments,
    validate_hierarchy_consistency,
    validate_section_references,
)
from .versioning import (
    AmendmentHistory,
    AmendmentVersion,
    generate_amended_version,
    generate_version_filename,
    save_version,
)

__all__ = [
    # Gazette conversion
    "build_gazette_chain",
    "convert_gazette_text",
    "convert_gazette_page_with_vision",
    "convert_gazette_pages_with_vision",
    "merge_gazette_pages",
    # Amendment extraction from PDFs
    "extract_amendments_from_pdf",
    "extract_section_context",
    "generate_registry_csv",
    "generate_details_tsv",
    # Footnote linking
    "FootnoteContext",
    "extract_markers_with_context",
    "extract_marker_from_annotation",
    "build_page_section_map",
    "link_footnotes_to_amendments",
    "validate_section_linkages",
    "ExtractedAmendment",
    "AmendmentExtractionResult",
    # Vision-based amendment extraction
    "VisionExtractedAmendment",
    # Vision-based extraction
    "VisionAmendmentExtractor",
    "ExtractionConfig",
    "extract_amendments_with_vision",
    # Vision cache
    "VisionExtractionCache",
    "get_default_cache",
    "clear_pdf_cache",
    "get_cache_stats",
    # Hybrid extraction
    "HybridAmendmentExtractor",
    "HybridExtractionConfig",
    "HybridExtractionResult",
    "extract_amendments_hybrid",
    # Evaluation
    "ExtractionEvaluator",
    "EvaluationReport",
    "evaluate_extraction_methods",
    # Amendment registry
    "AmendmentRecord",
    "AmendmentDetail",
    "parse_csv_registry",
    "parse_tsv_details",
    "validate_amendment_records",
    "group_amendments_by_act",
    "get_amendment_timeline",
    # Amendment application
    "apply_amendment",
    "apply_multiple_amendments",
    "extract_section_from_markup",
    "validate_amended_markup",
    # Validation
    "check_amendment_consistency",
    "detect_conflicting_amendments",
    "validate_section_references",
    "validate_hierarchy_consistency",
    # Diff generation
    "generate_text_diff",
    "generate_section_diff",
    "generate_summary_report",
    # Versioning
    "AmendmentHistory",
    "AmendmentVersion",
    "generate_version_filename",
    "save_version",
    "generate_amended_version",
]
