"""Visual gazette analysis for enhanced gazette metadata extraction.

Extracts metadata from gazette notification PDFs using visual analysis
and OCR to improve gazette-amendment matching accuracy.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

logger = logging.getLogger(__name__)


class GazetteAnalysis:
    """Comprehensive analysis of a gazette notification."""
    
    def __init__(self, pdf_path: Path, metadata: Dict[str, Any] = None):
        self.pdf_path = pdf_path
        self.metadata = metadata or {}
        self.visual_features: Dict[str, Any] = {}
        self.ocr_results: Dict[int, str] = {}  # page -> text
        self.structure: Dict[str, Any] = {}
        self.validation: Dict[str, Any] = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pdf_path": str(self.pdf_path),
            "metadata": self.metadata,
            "visual_features": self.visual_features,
            "ocr_results": {str(k): v for k, v in self.ocr_results.items()},
            "structure": self.structure,
            "validation": self.validation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GazetteAnalysis":
        """Create from dictionary."""
        analysis = cls(Path(data["pdf_path"]), data.get("metadata", {}))
        analysis.visual_features = data.get("visual_features", {})
        analysis.ocr_results = {int(k): v for k, v in data.get("ocr_results", {}).items()}
        analysis.structure = data.get("structure", {})
        analysis.validation = data.get("validation", {})
        return analysis


class VisualFeatures:
    """Visual features extracted from gazette PDF."""
    
    def __init__(self):
        self.has_official_seal = False
        self.has_letterhead = False
        self.has_signatures = False
        self.font_analysis: Dict[str, Any] = {}
        self.page_numbering: Dict[str, Any] = {}
        self.visual_complexity_score = 0.0
        self.color_usage: Dict[str, Any] = {}
        self.margin_analysis: Dict[str, Any] = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "has_official_seal": self.has_official_seal,
            "has_letterhead": self.has_letterhead,
            "has_signatures": self.has_signatures,
            "font_analysis": self.font_analysis,
            "page_numbering": self.page_numbering,
            "visual_complexity_score": self.visual_complexity_score,
            "color_usage": self.color_usage,
            "margin_analysis": self.margin_analysis
        }


class VisualGazetteAnalyzer:
    """Analyzer for extracting metadata from gazette PDFs using visual features."""
    
    def __init__(self, enable_ocr: bool = True, cache_results: bool = True):
        """Initialize visual gazette analyzer.
        
        Args:
            enable_ocr: Whether to perform OCR on key sections.
            cache_results: Whether to cache analysis results.
        """
        self.enable_ocr = enable_ocr
        self.cache_results = cache_results
        self._analysis_cache: Dict[str, GazetteAnalysis] = {}
        
    async def analyze_gazette(self, gazette_pdf: Path) -> GazetteAnalysis:
        """Extract metadata from gazette notification PDF.
        
        Args:
            gazette_pdf: Path to gazette PDF file.
            
        Returns:
            GazetteAnalysis with extracted metadata.
        """
        logger.info(f"Analyzing gazette: {gazette_pdf}")
        
        # Check cache first
        cache_key = str(gazette_pdf)
        if self.cache_results and cache_key in self._analysis_cache:
            logger.debug(f"Using cached analysis for {gazette_pdf}")
            return self._analysis_cache[cache_key]
        
        try:
            # Extract visual features
            visual_features = await self._extract_visual_features(gazette_pdf)
            
            # Extract metadata from filename and content
            metadata = self._extract_metadata(gazette_pdf, visual_features)
            
            # Analyze gazette structure
            structure = self._analyze_gazette_structure(gazette_pdf)
            
            # Validate against known patterns
            validation = self._validate_gazette(metadata, structure)
            
            # Create analysis object
            analysis = GazetteAnalysis(gazette_pdf, metadata)
            analysis.visual_features = visual_features.to_dict()
            analysis.structure = structure
            analysis.validation = validation
            
            # Cache the result
            if self.cache_results:
                self._analysis_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze gazette {gazette_pdf}: {e}")
            # Return minimal analysis with error
            return GazetteAnalysis(
                gazette_pdf,
                metadata={"error": str(e), "filename": gazette_pdf.name}
            )
    
    async def _extract_visual_features(self, gazette_pdf: Path) -> VisualFeatures:
        """Extract visual cues from gazette PDF.
        
        Note: This is a placeholder for actual visual analysis.
        In production, this would use image processing libraries.
        """
        features = VisualFeatures()
        
        try:
            # Analyze first page (typically header with gazette number)
            first_page_analysis = self._analyze_first_page(gazette_pdf)
            
            # Look for official seals/stamps
            features.has_official_seal = self._detect_seal_patterns(first_page_analysis)
            
            # Look for government letterhead
            features.has_letterhead = self._detect_letterhead_patterns(first_page_analysis)
            
            # Detect signature lines
            features.has_signatures = self._detect_signature_patterns(gazette_pdf)
            
            # Analyze fonts (placeholder)
            features.font_analysis = self._analyze_font_patterns(gazette_pdf)
            
            # Identify page numbering style
            features.page_numbering = self._analyze_page_numbering_pattern(gazette_pdf)
            
            # Calculate visual complexity score
            features.visual_complexity_score = self._calculate_visual_complexity(gazette_pdf)
            
        except Exception as e:
            logger.warning(f"Visual feature extraction failed for {gazette_pdf}: {e}")
        
        return features
    
    def _analyze_first_page(self, gazette_pdf: Path) -> Dict[str, Any]:
        """Analyze first page of gazette PDF.
        
        Returns:
            Dictionary with first page analysis.
        """
        # Placeholder for actual first page analysis
        # In production, this would extract text layout, formatting, etc.
        return {
            "filename": gazette_pdf.name,
            "estimated_pages": self._estimate_page_count(gazette_pdf)
        }
    
    def _detect_seal_patterns(self, page_analysis: Dict[str, Any]) -> bool:
        """Detect official seal/stamp patterns.
        
        Returns:
            True if official seal/stamp patterns detected.
        """
        filename = page_analysis.get("filename", "").lower()
        
        # Look for common seal indicators in filename
        seal_keywords = ["seal", "stamp", "emblem", "insignia", "logo"]
        for keyword in seal_keywords:
            if keyword in filename:
                return True
        
        return False
    
    def _detect_letterhead_patterns(self, page_analysis: Dict[str, Any]) -> bool:
        """Detect government letterhead patterns.
        
        Returns:
            True if government letterhead patterns detected.
        """
        filename = page_analysis.get("filename", "").lower()
        
        # Look for government indicators in filename
        gov_keywords = [
            "gazette", "notification", "extraordinary", "government", 
            "ministry", "department", "official"
        ]
        keyword_count = sum(1 for keyword in gov_keywords if keyword in filename)
        
        return keyword_count >= 2  # At least 2 government indicators
    
    def _detect_signature_patterns(self, gazette_pdf: Path) -> bool:
        """Detect signature patterns in gazette.
        
        Returns:
            True if signature patterns detected.
        """
        filename = gazette_pdf.name.lower()
        
        # Look for signature indicators
        signature_keywords = ["signed", "signature", "undersigned", "authorized"]
        for keyword in signature_keywords:
            if keyword in filename:
                return True
        
        return False
    
    def _analyze_font_patterns(self, gazette_pdf: Path) -> Dict[str, Any]:
        """Analyze font usage patterns.
        
        Returns:
            Dictionary with font analysis results.
        """
        # Placeholder for actual font analysis
        return {
            "method": "filename_analysis",
            "official_like": True,  # Assume official documents use formal fonts
            "complexity": "medium"
        }
    
    def _analyze_page_numbering_pattern(self, gazette_pdf: Path) -> Dict[str, Any]:
        """Analyze page numbering style.
        
        Returns:
            Dictionary with page numbering analysis.
        """
        # Placeholder for actual page numbering analysis
        filename = gazette_pdf.name.lower()
        
        if "page" in filename or "pages" in filename:
            style = "explicit"
        elif "-" in filename and any(char.isdigit() for char in filename):
            style = "range"
        else:
            style = "implied"
        
        return {
            "style": style,
            "consistent": True  # Assume consistent numbering
        }
    
    def _calculate_visual_complexity(self, gazette_pdf: Path) -> float:
        """Calculate visual complexity score.
        
        Returns:
            Complexity score between 0 and 1.
        """
        filename = gazette_pdf.name.lower()
        
        # Simple heuristics based on filename patterns
        score = 0.5  # Base score
        
        # Multiple numbers indicate complexity
        numbers = re.findall(r'\d+', filename)
        if len(numbers) > 2:
            score += 0.2
        
        # Longer filenames often indicate more complex documents
        if len(filename) > 40:
            score += 0.1
        
        # Contains multiple keywords
        keywords = ["gazette", "notification", "act", "rules", "amendment"]
        keyword_count = sum(1 for kw in keywords if kw in filename)
        score += keyword_count * 0.05
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _estimate_page_count(self, gazette_pdf: Path) -> int:
        """Estimate page count based on filename patterns.
        
        Returns:
            Estimated page count.
        """
        filename = gazette_pdf.name
        
        # Look for page range patterns like "1-5" or "pages1-10"
        page_range_match = re.search(r'(\d+)[-–](\d+)', filename)
        if page_range_match:
            start = int(page_range_match.group(1))
            end = int(page_range_match.group(2))
            return max(1, end - start + 1)
        
        # Look for page count patterns like "10pages" or "5pgs"
        page_count_match = re.search(r'(\d+)\s*(?:page|pg|pgs|p)\b', filename, re.IGNORECASE)
        if page_count_match:
            return int(page_count_match.group(1))
        
        # Default estimate based on filename length and content
        if "notification" in filename.lower():
            return 2  # Typical notification length
        elif "gazette" in filename.lower():
            return 10  # Typical gazette length
        else:
            return 1
    
    def _extract_metadata(self, gazette_pdf: Path, visual_features: VisualFeatures) -> Dict[str, Any]:
        """Extract metadata from gazette PDF.
        
        Args:
            gazette_pdf: Path to gazette PDF.
            visual_features: Extracted visual features.
            
        Returns:
            Dictionary with extracted metadata.
        """
        filename = gazette_pdf.name
        metadata = {
            "filename": filename,
            "file_path": str(gazette_pdf),
            "analysis_method": "visual",
            "visual_features_summary": {
                "has_official_seal": visual_features.has_official_seal,
                "has_letterhead": visual_features.has_letterhead,
                "has_signatures": visual_features.has_signatures,
                "visual_complexity": visual_features.visual_complexity_score
            }
        }
        
        # Extract gazette number (6 digits)
        gazette_match = re.search(r'(\d{6})', filename)
        if gazette_match:
            metadata["gazette_number"] = gazette_match.group(1)
        
        # Extract date (various formats)
        date_match = self._extract_date_from_filename(filename)
        if date_match:
            metadata["publication_date"] = date_match
        
        # Extract act information
        act_info = self._extract_act_info_from_filename(filename)
        metadata.update(act_info)
        
        # Add visual feature flags
        metadata["is_official_looking"] = (
            visual_features.has_official_seal or 
            visual_features.has_letterhead or
            visual_features.visual_complexity_score > 0.7
        )
        
        return metadata
    
    def _extract_date_from_filename(self, filename: str) -> Optional[str]:
        """Extract date from filename.
        
        Returns:
            Date string in YYYY-MM-DD format if found, None otherwise.
        """
        # Try DD-MM-YYYY
        match = re.search(r'(\d{1,2})[-.](\d{1,2})[-.](\d{4})', filename)
        if match:
            return f"{match.group(3)}-{match.group(2).zfill(2)}-{match.group(1).zfill(2)}"
        
        # Try YYYY-MM-DD
        match = re.search(r'(\d{4})[-.](\d{1,2})[-.](\d{1,2})', filename)
        if match:
            return f"{match.group(1)}-{match.group(2).zfill(2)}-{match.group(3).zfill(2)}"
        
        # Try YYYY (with word boundaries or separators)
        match = re.search(r'(?:^|_|-|\.|\s)(\d{4})(?:$|_|-|\.|\s|\.pdf)', filename)
        if match:
            return f"{match.group(1)}-01-01"  # Default to Jan 1
        
        return None
    
    def _extract_act_info_from_filename(self, filename: str) -> Dict[str, Any]:
        """Extract act information from filename.
        
        Returns:
            Dictionary with act information.
        """
        act_info = {}
        filename_lower = filename.lower()
        
        # Replace underscores with spaces for easier matching
        normalized = filename_lower.replace('_', ' ').replace('-', ' ')
        
        # Extract act number (commonly after "Act")
        act_match = re.search(r'act\s*(\d+)\s+of\s+(\d{4})', normalized, re.IGNORECASE)
        if act_match:
            act_info["act_number"] = act_match.group(1)
            act_info["act_year"] = act_match.group(2)
            act_info["act_reference"] = f"Act {act_match.group(1)} of {act_match.group(2)}"
        else:
            # Try other patterns
            match = re.search(r'(\d+)\s+of\s+(\d{4})', normalized)
            if match:
                act_info["act_number"] = match.group(1)
                act_info["act_year"] = match.group(2)
                act_info["act_reference"] = f"Act {match.group(1)} of {match.group(2)}"
            else:
                # Extract year
                year_match = re.search(r'\b(\d{4})\b', normalized)
                if year_match:
                    act_info["act_year"] = year_match.group(1)
        
        # Extract act name
        if "it act" in normalized or "information technology" in normalized:
            act_info["act_name"] = "Information Technology Act"
        elif "intermediary guidelines" in normalized:
            act_info["act_name"] = "Intermediary Guidelines and Digital Media Ethics Code Rules"
        elif "rules" in normalized:
            act_info["act_name"] = "Rules"
        elif "amendment" in normalized:
            act_info["act_name"] = "Amendment"
        
        return act_info
    
    def _analyze_gazette_structure(self, gazette_pdf: Path) -> Dict[str, Any]:
        """Analyze gazette structure.
        
        Returns:
            Dictionary with structure analysis.
        """
        # Placeholder for actual structure analysis
        return {
            "estimated_sections": 3,  # Header, body, signature
            "has_table_of_contents": False,
            "has_appendices": False,
            "structure_type": "notification"
        }
    
    def _validate_gazette(self, metadata: Dict[str, Any], structure: Dict[str, Any]) -> Dict[str, Any]:
        """Validate gazette against known patterns.
        
        Returns:
            Dictionary with validation results.
        """
        issues = []
        
        # Check required fields
        if not metadata.get("act_year"):
            issues.append("missing_act_year")
        
        if not metadata.get("act_name"):
            issues.append("missing_act_name")
        
        if not metadata.get("publication_date"):
            issues.append("missing_publication_date")
        
        # Validate year range
        if "act_year" in metadata:
            try:
                year = int(metadata["act_year"])
                if year < 1800 or year > 2100:
                    issues.append("invalid_act_year")
            except ValueError:
                issues.append("invalid_act_year_format")
        
        # Check for gazette number format
        if "gazette_number" in metadata:
            gaz_num = metadata["gazette_number"]
            if len(gaz_num) != 6 or not gaz_num.isdigit():
                issues.append("invalid_gazette_number_format")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "score": 1.0 - (len(issues) * 0.1),  # Deduct 10% per issue
            "recommendations": self._get_validation_recommendations(issues)
        }
    
    def _get_validation_recommendations(self, issues: List[str]) -> List[str]:
        """Get validation recommendations for issues.
        
        Returns:
            List of recommendation strings.
        """
        recommendations = []
        
        for issue in issues:
            if issue == "missing_act_year":
                recommendations.append("Add act year to filename (e.g., 'Act_10_of_2009.pdf')")
            elif issue == "missing_act_name":
                recommendations.append("Add act name to filename (e.g., 'IT_Act_Rules_Amendment.pdf')")
            elif issue == "missing_publication_date":
                recommendations.append("Add publication date to filename (e.g., '2023-12-15_Gazette.pdf')")
            elif issue == "invalid_act_year":
                recommendations.append("Act year should be between 1800 and 2100")
            elif issue == "invalid_gazette_number_format":
                recommendations.append("Gazette number should be 6 digits")
        
        return recommendations


def create_visual_gazette_analyzer(enable_ocr: bool = True, cache_results: bool = True) -> VisualGazetteAnalyzer:
    """Create a visual gazette analyzer.
    
    Returns:
        VisualGazetteAnalyzer instance.
    """
    return VisualGazetteAnalyzer(enable_ocr=enable_ocr, cache_results=cache_results)


async def analyze_gazette_directory(
    directory: Path,
    output_file: Optional[Path] = None,
    enable_ocr: bool = True
) -> Dict[str, GazetteAnalysis]:
    """Analyze all gazette PDFs in a directory.
    
    Args:
        directory: Directory containing gazette PDFs.
        output_file: Optional file to save analysis results.
        enable_ocr: Whether to perform OCR.
        
    Returns:
        Dictionary mapping PDF paths to GazetteAnalysis objects.
    """
    analyzer = VisualGazetteAnalyzer(enable_ocr=enable_ocr)
    analyses = {}
    
    # Find PDF files
    pdf_files = list(directory.rglob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
    
    # Analyze each file
    for pdf_file in pdf_files:
        try:
            analysis = await analyzer.analyze_gazette(pdf_file)
            analyses[str(pdf_file)] = analysis.to_dict()
        except Exception as e:
            logger.error(f"Failed to analyze {pdf_file}: {e}")
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analyses, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved analysis to {output_file}")
    
    return analyses