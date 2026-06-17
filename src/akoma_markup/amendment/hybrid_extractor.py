"""Hybrid amendment extraction combining vision and regex methods.

Provides fallback mechanism and intelligent combination of results
from both extraction approaches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import asyncio

from .extract import extract_amendments_from_pdf, AmendmentExtractionResult as RegexResult
from .vision_extractor import VisionAmendmentExtractor, ExtractionConfig
from .vision_schema import VisionExtractedAmendment, AmendmentExtractionResult as VisionResult
from .error_handling import (
    ErrorHandler, ErrorCategory, ErrorSeverity,
    handle_vision_extraction_error, handle_regex_extraction_error
)


logger = logging.getLogger(__name__)


@dataclass
class HybridExtractionConfig:
    """Configuration for hybrid extraction."""
    
    # Which methods to use
    use_vision: bool = True
    use_regex: bool = True
    
    # Vision extraction configuration
    vision_config: Optional[Dict[str, Any]] = None
    
    # Combination strategy
    confidence_threshold: float = 0.5
    prefer_vision: bool = True  # When both methods find same amendment, which to prefer
    require_agreement: bool = False  # Only include amendments both methods agree on
    
    # Fallback settings
    vision_fallback_to_regex: bool = True  # Use regex if vision fails
    regex_fallback_to_vision: bool = False  # Use vision if regex fails (less common)
    
    # Output settings
    include_source_method: bool = True  # Include which method found each amendment


@dataclass
class HybridExtractionResult:
    """Result of hybrid extraction."""
    
    pdf_path: str
    amendments: List[Any]  # Mixed: VisionExtractedAmendment or ExtractedAmendment
    vision_result: Optional[VisionResult] = None
    regex_result: Optional[RegexResult] = None
    combination_stats: Dict[str, Any] = None
    errors: List[Dict[str, Any]] = None
    error_summary: Dict[str, Any] = None
    
    @property
    def total_amendments(self) -> int:
        return len(self.amendments)
    
    @property
    def vision_count(self) -> int:
        if not self.combination_stats:
            return 0
        return self.combination_stats.get("vision_only", 0) + self.combination_stats.get("both_methods", 0)
    
    @property
    def regex_count(self) -> int:
        if not self.combination_stats:
            return 0
        return self.combination_stats.get("regex_only", 0) + self.combination_stats.get("both_methods", 0)


class HybridAmendmentExtractor:
    """Extractor that combines vision and regex methods."""
    
    def __init__(self, config: Optional[HybridExtractionConfig] = None):
        """Initialize hybrid extractor.
        
        Args:
            config: Hybrid extraction configuration.
        """
        self.config = config or HybridExtractionConfig()
        self.vision_extractor = None
        
        # Initialize error handler with empty config
        error_config = {}
        self.error_handler = ErrorHandler(error_config)
        
        if self.config.use_vision:
            vision_config = self.config.vision_config or {}
            self.vision_extractor = VisionAmendmentExtractor(vision_config)
    
    async def extract(
        self, 
        pdf_path: Path,
        page_range: Optional[range] = None
    ) -> HybridExtractionResult:
        """Extract amendments using hybrid approach.
        
        Args:
            pdf_path: Path to PDF file.
            page_range: Optional range of pages to process.
            
        Returns:
            HybridExtractionResult with combined amendments.
        """
        logger.info(f"Starting hybrid extraction from {pdf_path}")
        
        vision_result = None
        regex_result = None
        
        # Run vision extraction if enabled
        if self.config.use_vision and self.vision_extractor:
            try:
                vision_result = await self.vision_extractor.extract_from_pdf(pdf_path, page_range)
                logger.info(f"Vision extraction found {len(vision_result.extracted_amendments)} amendments")
            except Exception as e:
                # Use enhanced error handling
                handle_vision_extraction_error(self.error_handler, e, pdf_path, page_range)
                logger.error(f"Vision extraction failed: {e}")
                if not self.config.vision_fallback_to_regex:
                    raise
                vision_result = None
        
        # Run regex extraction if enabled
        if self.config.use_regex:
            try:
                regex_result = extract_amendments_from_pdf(pdf_path)
                logger.info(f"Regex extraction found {len(regex_result.amendments)} amendments")
            except Exception as e:
                # Use enhanced error handling
                handle_regex_extraction_error(self.error_handler, e, pdf_path)
                logger.error(f"Regex extraction failed: {e}")
                if not self.config.regex_fallback_to_vision and not vision_result:
                    raise
                regex_result = None
        
        # Combine results
        combined_amendments, stats = self.combine_results(vision_result, regex_result)
        
        # Apply confidence threshold
        filtered_amendments = self._filter_by_confidence(combined_amendments)
        
        logger.info(f"Hybrid extraction complete: {len(filtered_amendments)} amendments "
                   f"(vision: {stats.get('vision_only', 0)}, "
                   f"regex: {stats.get('regex_only', 0)}, "
                   f"both: {stats.get('both_methods', 0)})")
        
        # Create error report
        error_report = self.error_handler.create_error_report(pdf_path)
        
        return HybridExtractionResult(
            pdf_path=str(pdf_path),
            amendments=filtered_amendments,
            vision_result=vision_result,
            regex_result=regex_result,
            combination_stats=stats,
            errors=[e.to_dict() for e in self.error_handler.errors],
            error_summary=error_report["error_summary"]
        )
    
    def combine_results(
        self, 
        vision_result: Optional[VisionResult], 
        regex_result: Optional[RegexResult]
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Combine results from vision and regex extraction.
        
        Args:
            vision_result: Vision extraction result.
            regex_result: Regex extraction result.
            
        Returns:
            Tuple of (combined amendments, combination statistics).
        """
        vision_amendments = []
        regex_amendments = []
        
        if vision_result:
            vision_amendments = vision_result.extracted_amendments
        
        if regex_result:
            regex_amendments = regex_result.amendments
        
        # Convert to comparable format for deduplication
        vision_keys = self._extract_amendment_keys(vision_amendments, "vision")
        regex_keys = self._extract_amendment_keys(regex_amendments, "regex")
        
        # Find overlaps and unique amendments
        vision_key_set = set(vision_keys.keys())
        regex_key_set = set(regex_keys.keys())
        
        common_keys = vision_key_set.intersection(regex_key_set)
        vision_only_keys = vision_key_set - regex_key_set
        regex_only_keys = regex_key_set - vision_key_set
        
        # Combine amendments
        combined = []
        
        # Handle amendments found by both methods
        for key in common_keys:
            vision_am = vision_keys[key]
            regex_am = regex_keys[key]
            
            if self.config.prefer_vision:
                selected = vision_am
                source = "vision+regex" if self.config.include_source_method else "both"
            else:
                selected = regex_am
                source = "vision+regex" if self.config.include_source_method else "both"
            
            # Add source metadata if requested
            if self.config.include_source_method:
                selected = self._add_source_metadata(selected, source)
            
            combined.append(selected)
        
        # Handle vision-only amendments
        for key in vision_only_keys:
            amendment = vision_keys[key]
            if self.config.include_source_method:
                amendment = self._add_source_metadata(amendment, "vision")
            combined.append(amendment)
        
        # Handle regex-only amendments
        for key in regex_only_keys:
            amendment = regex_keys[key]
            if self.config.include_source_method:
                amendment = self._add_source_metadata(amendment, "regex")
            combined.append(amendment)
        
        # Statistics
        stats = {
            "vision_total": len(vision_amendments),
            "regex_total": len(regex_amendments),
            "vision_only": len(vision_only_keys),
            "regex_only": len(regex_only_keys),
            "both_methods": len(common_keys),
            "combined_total": len(combined),
        }
        
        return combined, stats
    
    def resolve_conflicts(self, conflicting_amendments: List[Tuple[Any, Any]]) -> List[Any]:
        """Resolve conflicts between amendments from different methods.
        
        Args:
            conflicting_amendments: List of (vision_amendment, regex_amendment) pairs.
            
        Returns:
            List of resolved amendments.
        """
        resolved = []
        
        for vision_am, regex_am in conflicting_amendments:
            # Default strategy: prefer higher confidence
            vision_conf = getattr(vision_am, "confidence_score", 0.5)
            regex_conf = getattr(regex_am, "linkage_confidence", "none")
            regex_conf_value = {"high": 0.9, "medium": 0.7, "low": 0.5, "none": 0.3}.get(regex_conf, 0.3)
            
            if vision_conf >= regex_conf_value:
                selected = vision_am
                source = "vision"
            else:
                selected = regex_am
                source = "regex"
            
            # Add conflict resolution metadata
            if hasattr(selected, "metadata"):
                selected.metadata = selected.metadata or {}
                selected.metadata["conflict_resolution"] = {
                    "method": "confidence_comparison",
                    "selected_source": source,
                    "vision_confidence": vision_conf,
                    "regex_confidence": regex_conf,
                }
            
            resolved.append(selected)
        
        return resolved
    
    def _extract_amendment_keys(self, amendments: List[Any], source: str) -> Dict[str, Any]:
        """Extract unique keys for amendment deduplication.
        
        Args:
            amendments: List of amendments.
            source: Source method ("vision" or "regex").
            
        Returns:
            Dictionary mapping keys to amendments.
        """
        keys = {}
        
        for i, amendment in enumerate(amendments):
            # Extract key fields based on amendment type
            if source == "vision":
                # VisionExtractedAmendment
                key = self._vision_amendment_key(amendment)
            else:
                # ExtractedAmendment (regex)
                key = self._regex_amendment_key(amendment)
            
            if key:
                keys[key] = amendment
        
        return keys
    
    def _vision_amendment_key(self, amendment: VisionExtractedAmendment) -> Optional[str]:
        """Create unique key for vision amendment."""
        try:
            return (
                f"{amendment.page_num}:"
                f"{amendment.act_number or ''}:"
                f"{amendment.act_year or ''}:"
                f"{amendment.section_number or ''}:"
                f"{amendment.amendment_type or ''}:"
                f"{amendment.target_location or ''}:"
                f"{amendment.original_text[:100] if amendment.original_text else ''}"
            )
        except AttributeError:
            return None
    
    def _regex_amendment_key(self, amendment: Any) -> Optional[str]:
        """Create unique key for regex amendment."""
        try:
            # Try to extract similar fields as vision amendment
            page_num = getattr(amendment, "page_num", getattr(amendment, "source_page", 0))
            act_number = getattr(amendment, "act_number", getattr(amendment, "amendment_act_id", ""))
            section = getattr(amendment, "target_section", getattr(amendment, "section", ""))
            amendment_type = getattr(amendment, "amendment_type", "")
            text = getattr(amendment, "original_text", getattr(amendment, "text", ""))
            
            return (
                f"{page_num}:"
                f"{act_number or ''}:"
                f"{section or ''}:"
                f"{amendment_type or ''}:"
                f"{text[:100] if text else ''}"
            )
        except AttributeError:
            return None
    
    def _add_source_metadata(self, amendment: Any, source: str) -> Any:
        """Add source metadata to amendment.
        
        Args:
            amendment: Amendment object.
            source: Source method ("vision", "regex", or "vision+regex").
            
        Returns:
            Amendment with added metadata.
        """
        # For VisionExtractedAmendment
        if hasattr(amendment, "raw_llm_response"):
            if not hasattr(amendment, "metadata"):
                amendment.metadata = {}
            amendment.metadata["extraction_source"] = source
        # For ExtractedAmendment (regex)
        elif hasattr(amendment, "to_dict"):
            if not hasattr(amendment, "metadata"):
                amendment.metadata = {}
            amendment.metadata["extraction_source"] = source
        
        return amendment
    
    def _filter_by_confidence(self, amendments: List[Any]) -> List[Any]:
        """Filter amendments by confidence threshold.
        
        Args:
            amendments: List of amendments.
            
        Returns:
            Filtered list of amendments.
        """
        if self.config.confidence_threshold <= 0:
            return amendments
        
        filtered = []
        
        for amendment in amendments:
            # Get confidence score based on amendment type
            confidence = 1.0  # Default high confidence
            
            # Vision amendment
            if hasattr(amendment, "confidence_score"):
                confidence = amendment.confidence_score
            # Regex amendment
            elif hasattr(amendment, "linkage_confidence"):
                conf_map = {"high": 0.9, "medium": 0.7, "low": 0.5, "none": 0.3}
                confidence = conf_map.get(amendment.linkage_confidence, 0.3)
            
            if confidence >= self.config.confidence_threshold:
                filtered.append(amendment)
            else:
                logger.debug(f"Filtered out low-confidence amendment: {confidence}")
        
        return filtered


# Convenience function
async def extract_amendments_hybrid(
    pdf_path: Path,
    config: Optional[Dict[str, Any]] = None,
    page_range: Optional[range] = None
) -> HybridExtractionResult:
    """Convenience function for hybrid amendment extraction.
    
    Args:
        pdf_path: Path to PDF file.
        config: Optional hybrid extraction configuration.
        page_range: Optional range of pages to process.
        
    Returns:
        HybridExtractionResult.
    """
    if config is None:
        config = {}
    
    hybrid_config = HybridExtractionConfig(**config)
    extractor = HybridAmendmentExtractor(hybrid_config)
    return await extractor.extract(pdf_path, page_range)