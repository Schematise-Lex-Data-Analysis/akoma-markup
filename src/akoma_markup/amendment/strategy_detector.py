"""Strategy detection for hybrid amendment extraction.

Analyzes PDF characteristics to determine optimal extraction strategy:
- vision_only: For high-quality scans, dense amendments, or gazette notifications
- regex_only: For poor quality scans
- hybrid_parallel: For complex layouts
- hybrid_sequential: Default balanced approach
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Literal

import pdfplumber

logger = logging.getLogger(__name__)


class ExtractionStrategy(str, Enum):
    """Extraction strategies for hybrid system."""
    VISION_ONLY = "vision_only"
    REGEX_ONLY = "regex_only"
    HYBRID_PARALLEL = "hybrid_parallel"
    HYBRID_SEQUENTIAL = "hybrid_sequential"
    AUTO = "auto"


@dataclass
class PDFCharacteristics:
    """Characteristics of a PDF for strategy determination."""
    
    page_count: int
    scan_quality: float  # 0.0-1.0, higher is better
    amendment_density: float  # 0.0-1.0, estimated density of amendments
    complex_layout: bool  # True if PDF has complex multi-column layout
    gazette_notification: bool  # True if appears to be a gazette notification
    estimated_cost: float  # Estimated processing cost


class ExtractionStrategyDetector:
    """Determines optimal extraction strategy based on PDF characteristics."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize strategy detector.
        
        Args:
            config: Configuration dictionary with strategy thresholds.
        """
        self.config = config or {}
        self.scan_quality_threshold = self.config.get("scan_quality_threshold", 0.7)
        self.amendment_density_threshold = self.config.get("amendment_density_threshold", 0.8)
        self.complex_layout_threshold = self.config.get("complex_layout_threshold", 0.6)
    
    def determine_strategy(self, pdf_path: Path) -> ExtractionStrategy:
        """Determine optimal extraction strategy for PDF.
        
        Args:
            pdf_path: Path to PDF file.
            
        Returns:
            ExtractionStrategy enum value.
        """
        logger.info(f"Analyzing PDF for strategy determination: {pdf_path}")
        
        # Analyze PDF characteristics
        characteristics = self._analyze_pdf(pdf_path)
        
        # Decision tree based on characteristics
        if characteristics.scan_quality < self.scan_quality_threshold:
            logger.info("Poor scan quality detected, using regex_only")
            return ExtractionStrategy.REGEX_ONLY
        
        if characteristics.amendment_density > self.amendment_density_threshold:
            logger.info("High amendment density detected, using vision_only")
            return ExtractionStrategy.VISION_ONLY
            
        if characteristics.complex_layout:
            logger.info("Complex layout detected, using hybrid_parallel")
            return ExtractionStrategy.HYBRID_PARALLEL
            
        if characteristics.gazette_notification:
            logger.info("Gazette notification detected, using vision_only")
            return ExtractionStrategy.VISION_ONLY
            
        logger.info("Defaulting to hybrid_sequential strategy")
        return ExtractionStrategy.HYBRID_SEQUENTIAL
    
    def _analyze_pdf(self, pdf_path: Path) -> PDFCharacteristics:
        """Analyze PDF to extract characteristics for strategy determination.
        
        Args:
            pdf_path: Path to PDF file.
            
        Returns:
            PDFCharacteristics object.
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                
                # Sample first few pages for analysis
                sample_pages = min(5, page_count)
                scan_quality = self._estimate_scan_quality(pdf_path, sample_pages)
                amendment_density = self._estimate_amendment_density(pdf_path, sample_pages)
                complex_layout = self._detect_complex_layout(pdf_path, sample_pages)
                gazette_notification = self._detect_gazette_format(pdf_path, sample_pages)
                estimated_cost = self._estimate_processing_cost(
                    page_count, scan_quality, amendment_density, complex_layout
                )
                
                return PDFCharacteristics(
                    page_count=page_count,
                    scan_quality=scan_quality,
                    amendment_density=amendment_density,
                    complex_layout=complex_layout,
                    gazette_notification=gazette_notification,
                    estimated_cost=estimated_cost
                )
        except Exception as e:
            logger.error(f"Failed to analyze PDF {pdf_path}: {e}")
            # Return default characteristics on error
            return PDFCharacteristics(
                page_count=1,
                scan_quality=0.8,  # Assume reasonable quality
                amendment_density=0.5,
                complex_layout=False,
                gazette_notification=False,
                estimated_cost=1.0
            )
    
    def _estimate_scan_quality(self, pdf_path: Path, sample_pages: int = 3) -> float:
        """Estimate scan quality of PDF (0.0-1.0).
        
        Higher values indicate better OCR/vision processing potential.
        """
        # Simplified implementation - check for common scan quality indicators
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_score = 0.0
                pages_checked = 0
                
                for i in range(min(sample_pages, len(pdf.pages))):
                    page = pdf.pages[i]
                    page_score = 0.0
                    
                    # Check 1: Text density (higher is better for regex)
                    text = page.extract_text() or ""
                    text_density = len(text) / 10000  # Normalized
                    page_score += min(text_density * 0.5, 0.5)
                    
                    # Check 2: Image content (lower is better for regex)
                    # In pdfplumber, images indicate scans
                    images = page.images
                    if not images:
                        page_score += 0.3  # No images = likely born-digital
                    
                    # Check 3: Font consistency (simplified)
                    # Could be enhanced with actual font analysis
                    
                    total_score += page_score
                    pages_checked += 1
                
                return total_score / pages_checked if pages_checked > 0 else 0.7
        except Exception as e:
            logger.warning(f"Failed to estimate scan quality: {e}")
            return 0.7  # Default assumption
    
    def _estimate_amendment_density(self, pdf_path: Path, sample_pages: int = 3) -> float:
        """Estimate density of amendments in PDF (0.0-1.0).
        
        Higher values indicate more amendments per page.
        """
        # Simplified implementation - look for amendment indicators
        try:
            with pdfplumber.open(pdf_path) as pdf:
                amendment_indicators = 0
                total_words = 0
                
                for i in range(min(sample_pages, len(pdf.pages))):
                    page = pdf.pages[i]
                    text = page.extract_text() or ""
                    words = text.split()
                    total_words += len(words)
                    
                    # Look for amendment-related keywords
                    amendment_keywords = [
                        "amendment", "insert", "delete", "replace", "section",
                        "subsection", "clause", "schedule", "act", "rule"
                    ]
                    
                    for word in words:
                        if any(keyword in word.lower() for keyword in amendment_keywords):
                            amendment_indicators += 1
                
                # Normalize: amendments per 100 words
                if total_words > 0:
                    density = (amendment_indicators / total_words) * 100
                    # Scale to 0-1 range (assuming >5 per 100 words is dense)
                    return min(density / 5.0, 1.0)
                return 0.5
        except Exception as e:
            logger.warning(f"Failed to estimate amendment density: {e}")
            return 0.5  # Default assumption
    
    def _detect_complex_layout(self, pdf_path: Path, sample_pages: int = 3) -> bool:
        """Detect if PDF has complex layout (multi-column, tables, etc.).
        
        Complex layouts benefit from vision processing.
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                complex_count = 0
                
                for i in range(min(sample_pages, len(pdf.pages))):
                    page = pdf.pages[i]
                    
                    # Check for tables (indicator of complex layout)
                    tables = page.find_tables()
                    if tables:
                        complex_count += 1
                        continue
                    
                    # Check for multi-column layout via text bounding boxes
                    chars = page.chars
                    if len(chars) > 50:  # Enough text to analyze
                        # Simplified: check if text spans multiple columns
                        x_positions = [char['x0'] for char in chars]
                        if len(set(int(x/100) for x in x_positions)) > 2:
                            complex_count += 1
                
                # If majority of sampled pages are complex
                return (complex_count / min(sample_pages, len(pdf.pages))) > self.complex_layout_threshold
        except Exception as e:
            logger.warning(f"Failed to detect complex layout: {e}")
            return False
    
    def _detect_gazette_format(self, pdf_path: Path, sample_pages: int = 3) -> bool:
        """Detect if PDF appears to be a gazette notification.
        
        Gazette notifications have specific visual formatting.
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Check first page for gazette indicators
                if len(pdf.pages) > 0:
                    first_page = pdf.pages[0]
                    text = first_page.extract_text() or ""
                    
                    # Look for gazette indicators
                    gazette_indicators = [
                        "extraordinary", "gazette", "notification", "government of india",
                        "ministry of", "department of", "published by authority"
                    ]
                    
                    text_lower = text.lower()
                    indicator_count = sum(1 for indicator in gazette_indicators 
                                        if indicator in text_lower)
                    
                    return indicator_count >= 2  # At least 2 indicators
        except Exception as e:
            logger.warning(f"Failed to detect gazette format: {e}")
            return False
    
    def _estimate_processing_cost(
        self, 
        page_count: int, 
        scan_quality: float,
        amendment_density: float,
        complex_layout: bool
    ) -> float:
        """Estimate relative processing cost for different strategies.
        
        Returns normalized cost estimate (higher = more expensive).
        """
        base_cost = page_count * 1.0
        
        # Adjust for scan quality (poor quality = more regex work)
        if scan_quality < 0.5:
            base_cost *= 1.5
        
        # Adjust for amendment density (dense = more processing)
        base_cost *= (1.0 + amendment_density)
        
        # Adjust for complex layout (complex = more vision work)
        if complex_layout:
            base_cost *= 1.3
        
        return base_cost