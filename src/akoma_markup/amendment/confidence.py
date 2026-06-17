"""Multi-factor confidence scoring for amendment extraction results.

Calculates comprehensive confidence scores based on multiple factors
including source reliability, field consistency, and contextual validation.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for scoring."""
    
    VERY_HIGH = 0.9
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    VERY_LOW = 0.2


@dataclass
class ConfidenceScores:
    """Comprehensive confidence scores for an amendment."""
    
    overall: float
    factors: Dict[str, float]
    weights: Dict[str, float]
    level: ConfidenceLevel
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if overall confidence is high."""
        return self.overall >= ConfidenceLevel.HIGH.value
    
    @property
    def needs_review(self) -> bool:
        """Check if amendment needs manual review."""
        return self.overall < ConfidenceLevel.MEDIUM.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall": self.overall,
            "level": self.level.name,
            "factors": self.factors,
            "weights": self.weights,
            "needs_review": self.needs_review,
            "recommendations": self.recommendations
        }


@dataclass
class ExtractionContext:
    """Context for confidence scoring."""
    
    pdf_path: str
    total_pages: int = 0
    extraction_method: str = ""
    gazette_references: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_gazette_context(self) -> bool:
        """Check if gazette context is available."""
        return len(self.gazette_references) > 0


class ConfidenceScorer:
    """Multi-factor confidence scorer for amendments."""
    
    # Default weights for confidence factors
    DEFAULT_WEIGHTS = {
        "source": 0.2,           # Source reliability
        "internal_consistency": 0.15,  # Internal field consistency
        "cross_modal": 0.2,      # Cross-modal agreement (vision vs regex)
        "pattern_match": 0.15,   # Pattern matching quality
        "contextual": 0.1,       # Contextual validation
        "gazette": 0.1,          # Gazette reference matching
        "temporal": 0.1,         # Temporal consistency
    }
    
    # Source confidence mapping
    SOURCE_CONFIDENCE = {
        "vision_only": 0.8,      # Vision extraction
        "regex_only": 0.6,       # Regex extraction
        "vision+regex": 0.9,     # Both methods agree
        "hybrid": 0.85,          # Hybrid extraction
        "manual": 0.95,          # Manual entry
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize confidence scorer.
        
        Args:
            weights: Optional custom weights for confidence factors.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Weights sum to {weight_sum}, normalizing to 1.0")
            for key in self.weights:
                self.weights[key] /= weight_sum
    
    def calculate_confidence(
        self, 
        amendment: Any, 
        context: ExtractionContext
    ) -> ConfidenceScores:
        """Calculate comprehensive confidence scores for an amendment.
        
        Args:
            amendment: Amendment object.
            context: Extraction context.
            
        Returns:
            ConfidenceScores object.
        """
        logger.debug(f"Calculating confidence for amendment in {context.pdf_path}")
        
        scores = {}
        recommendations = []
        
        # 1. Source confidence
        source_score = self._calculate_source_confidence(amendment)
        scores["source"] = source_score
        
        # 2. Internal consistency confidence
        internal_score = self._calculate_internal_consistency(amendment)
        scores["internal_consistency"] = internal_score
        
        if internal_score < 0.6:
            recommendations.append("Check internal field consistency")
        
        # 3. Cross-modal agreement confidence
        if hasattr(amendment, "vision_regex_agreement"):
            cross_modal_score = amendment.vision_regex_agreement
        else:
            cross_modal_score = self._estimate_cross_modal_agreement(amendment)
        scores["cross_modal"] = cross_modal_score
        
        # 4. Pattern match confidence
        pattern_score = self._calculate_pattern_match_confidence(amendment)
        scores["pattern_match"] = pattern_score
        
        if pattern_score < 0.5:
            recommendations.append("Pattern match quality low")
        
        # 5. Contextual confidence
        contextual_score = self._calculate_contextual_confidence(amendment, context)
        scores["contextual"] = contextual_score
        
        # 6. Gazette reference confidence
        gazette_score = self._calculate_gazette_confidence(amendment, context)
        scores["gazette"] = gazette_score
        
        if gazette_score < 0.3 and context.has_gazette_context:
            recommendations.append("Gazette reference matching low")
        
        # 7. Temporal confidence
        temporal_score = self._calculate_temporal_confidence(amendment)
        scores["temporal"] = temporal_score
        
        if temporal_score < 0.5:
            recommendations.append("Check temporal consistency")
        
        # Calculate weighted overall confidence
        overall = self._calculate_weighted_average(scores)
        
        # Determine confidence level
        level = self._determine_confidence_level(overall)
        
        return ConfidenceScores(
            overall=overall,
            factors=scores,
            weights=self.weights,
            level=level,
            recommendations=recommendations
        )
    
    def _calculate_source_confidence(self, amendment: Any) -> float:
        """Calculate source reliability confidence.
        
        Args:
            amendment: Amendment object.
            
        Returns:
            Source confidence score.
        """
        # Determine extraction source
        source = "regex_only"  # Default
        
        if hasattr(amendment, "metadata"):
            metadata = getattr(amendment, "metadata", {})
            extraction_source = metadata.get("extraction_source", "")
            
            if extraction_source:
                if "vision" in extraction_source and "regex" in extraction_source:
                    source = "vision+regex"
                elif "vision" in extraction_source:
                    source = "vision_only"
                elif "regex" in extraction_source:
                    source = "regex_only"
                elif "hybrid" in extraction_source:
                    source = "hybrid"
                elif "manual" in extraction_source:
                    source = "manual"
        
        # Get source confidence
        return self.SOURCE_CONFIDENCE.get(source, 0.5)
    
    def _calculate_internal_consistency(self, amendment: Any) -> float:
        """Calculate internal field consistency confidence.
        
        Args:
            amendment: Amendment object.
            
        Returns:
            Internal consistency score.
        """
        score = 1.0  # Start with perfect score
        
        # Extract fields
        fields = self._extract_fields(amendment)
        
        # Check 1: Required fields presence
        required_fields = ["section_number", "amendment_type"]
        missing_required = sum(1 for field in required_fields if not fields.get(field))
        if missing_required > 0:
            score -= 0.2 * missing_required
        
        # Check 2: Field format consistency
        format_issues = 0
        
        # Section number format
        section = fields.get("section_number")
        if section:
            if not re.match(r'^\d+[A-Z]?$', str(section)):
                format_issues += 1
        
        # Act year format
        year = fields.get("act_year")
        if year:
            try:
                year_int = int(str(year))
                if year_int < 1800 or year_int > 2100:
                    format_issues += 1
            except ValueError:
                format_issues += 1
        
        # Page number format
        page_num = fields.get("page_num")
        if page_num:
            try:
                page_int = int(str(page_num))
                if page_int <= 0:
                    format_issues += 1
            except ValueError:
                format_issues += 1
        
        if format_issues > 0:
            score -= 0.1 * format_issues
        
        # Check 3: Logical consistency
        # Example: Insert operation should have text to insert
        amendment_type = fields.get("amendment_type", "").lower()
        original_text = fields.get("original_text", "")
        
        if amendment_type in ["insert", "substitute", "replace"] and not original_text:
            score -= 0.2
        
        # Check 4: Field value ranges
        # Section number should be reasonable
        if section:
            try:
                section_num = int(re.sub(r'[A-Z]', '', str(section)))
                if section_num > 999:  # Unlikely to have section > 999
                    score -= 0.1
            except ValueError:
                pass
        
        return max(score, 0.1)  # Minimum 0.1
    
    def _estimate_cross_modal_agreement(self, amendment: Any) -> float:
        """Estimate cross-modal agreement confidence.
        
        Args:
            amendment: Amendment object.
            
        Returns:
            Cross-modal agreement score.
        """
        # Check for fusion metadata
        if hasattr(amendment, "metadata"):
            metadata = getattr(amendment, "metadata", {})
            fusion_info = metadata.get("fusion_info", {})
            
            if fusion_info:
                fusion_type = fusion_info.get("type", "")
                
                if fusion_type == "agreement":
                    return 0.9  # High confidence for agreement
                elif fusion_type == "single_source":
                    source = fusion_info.get("source", "")
                    if source == "vision":
                        return 0.7
                    elif source == "regex":
                        return 0.5
                elif fusion_type == "confidence_weighted":
                    # Use the confidence values from fusion
                    vision_conf = fusion_info.get("vision_confidence", 0.5)
                    regex_conf = fusion_info.get("regex_confidence", 0.5)
                    return (vision_conf + regex_conf) / 2
        
        # Default: assume single source regex extraction
        return 0.5
    
    def _calculate_pattern_match_confidence(self, amendment: Any) -> float:
        """Calculate pattern matching confidence.
        
        Args:
            amendment: Amendment object.
            
        Returns:
            Pattern match confidence score.
        """
        # Extract text for pattern analysis
        original_text = ""
        if hasattr(amendment, "original_text"):
            original_text = str(amendment.original_text)
        elif hasattr(amendment, "text"):
            original_text = str(amendment.text)
        
        if not original_text:
            return 0.3  # Low confidence without text
        
        score = 0.5  # Base score
        
        # Check for amendment pattern keywords
        amendment_keywords = [
            "insert", "delete", "substitute", "replace", "add", "omit",
            "shall be", "shall be deemed", "in section", "of the act",
            "after", "before", "for", "the following"
        ]
        
        keyword_matches = sum(1 for kw in amendment_keywords if kw.lower() in original_text.lower())
        score += min(keyword_matches * 0.05, 0.3)  # Max 0.3 boost
        
        # Check for legal citation patterns
        citation_patterns = [
            r'section\s+\d+[A-Z]?',  # Section reference
            r'Act\s+\d+\s+of\s+\d{4}',  # Act reference
            r'\(\d+\)',  # Subsection reference
            r'clause\s+\([a-z]\)',  # Clause reference
        ]
        
        pattern_matches = sum(1 for pattern in citation_patterns if re.search(pattern, original_text, re.IGNORECASE))
        score += min(pattern_matches * 0.05, 0.2)  # Max 0.2 boost
        
        # Check text length and structure
        words = original_text.split()
        if len(words) >= 10:  # Reasonable minimum for amendment text
            score += 0.1
        
        # Check for proper formatting
        if any(char in original_text for char in [";", ":", "(", ")"]):
            score += 0.1  # Structured text
        
        return min(score, 1.0)
    
    def _calculate_contextual_confidence(
        self, 
        amendment: Any, 
        context: ExtractionContext
    ) -> float:
        """Calculate contextual validation confidence.
        
        Args:
            amendment: Amendment object.
            context: Extraction context.
            
        Returns:
            Contextual confidence score.
        """
        score = 0.5  # Base score
        
        # Extract amendment fields
        fields = self._extract_fields(amendment)
        
        # Check 1: Page number within document bounds
        page_num = fields.get("page_num")
        if page_num and context.total_pages > 0:
            try:
                page_int = int(str(page_num))
                if 1 <= page_int <= context.total_pages:
                    score += 0.2
                else:
                    score -= 0.3
            except ValueError:
                pass
        
        # Check 2: Act year context (if gazette references available)
        if context.has_gazette_context:
            act_year = fields.get("act_year")
            if act_year:
                # Check if year matches any gazette context
                matching_years = sum(1 for gazette in context.gazette_references 
                                   if str(gazette.get("year", "")) == str(act_year))
                if matching_years > 0:
                    score += 0.2
                else:
                    # Check for year proximity
                    for gazette in context.gazette_references:
                        gaz_year = gazette.get("year")
                        if gaz_year:
                            try:
                                year_diff = abs(int(act_year) - int(gaz_year))
                                if year_diff <= 2:
                                    score += 0.1
                                    break
                            except ValueError:
                                pass
        
        # Check 3: Validation results from context
        validation_results = context.validation_results
        if validation_results:
            # Check if amendment passed validation
            amd_id = self._get_amendment_id(amendment)
            if amd_id in validation_results.get("passed", []):
                score += 0.2
            elif amd_id in validation_results.get("failed", []):
                score -= 0.3
        
        return max(score, 0.1)
    
    def _calculate_gazette_confidence(
        self, 
        amendment: Any, 
        context: ExtractionContext
    ) -> float:
        """Calculate gazette reference matching confidence.
        
        Args:
            amendment: Amendment object.
            context: Extraction context.
            
        Returns:
            Gazette confidence score.
        """
        if not context.has_gazette_context:
            return 0.3  # Low confidence without gazette context
        
        # Extract amendment fields
        fields = self._extract_fields(amendment)
        
        best_match_score = 0.0
        
        for gazette in context.gazette_references:
            match_score = self._calculate_gazette_match_score(fields, gazette)
            best_match_score = max(best_match_score, match_score)
        
        return best_match_score
    
    def _calculate_gazette_match_score(
        self, 
        amendment_fields: Dict[str, Any], 
        gazette: Dict[str, Any]
    ) -> float:
        """Calculate match score between amendment and gazette.
        
        Args:
            amendment_fields: Amendment field dictionary.
            gazette: Gazette reference dictionary.
            
        Returns:
            Match score between 0 and 1.
        """
        score = 0.0
        
        # Year match (most important)
        amd_year = amendment_fields.get("act_year")
        gaz_year = gazette.get("year")
        
        if amd_year and gaz_year and str(amd_year) == str(gaz_year):
            score += 0.5
        elif amd_year and gaz_year:
            try:
                year_diff = abs(int(amd_year) - int(gaz_year))
                if year_diff == 1:
                    score += 0.3
                elif year_diff == 2:
                    score += 0.1
            except ValueError:
                pass
        
        # Act number match
        amd_act_num = amendment_fields.get("act_number")
        gaz_act_num = gazette.get("act_number")
        
        if amd_act_num and gaz_act_num and str(amd_act_num) == str(gaz_act_num):
            score += 0.3
        elif amd_act_num and gaz_act_num:
            # Try to extract numbers
            amd_num = re.search(r'\d+', str(amd_act_num))
            gaz_num = re.search(r'\d+', str(gaz_act_num))
            if amd_num and gaz_num and amd_num.group() == gaz_num.group():
                score += 0.2
        
        # Gazette number match
        amd_gazette = amendment_fields.get("gazette_number")
        gaz_gazette = gazette.get("gazette_number")
        
        if amd_gazette and gaz_gazette and str(amd_gazette) == str(gaz_gazette):
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_temporal_confidence(self, amendment: Any) -> float:
        """Calculate temporal consistency confidence.
        
        Args:
            amendment: Amendment object.
            
        Returns:
            Temporal confidence score.
        """
        score = 0.5  # Base score
        
        # Extract effective date
        effective_date = None
        if hasattr(amendment, "effective_date"):
            effective_date = amendment.effective_date
        elif hasattr(amendment, "date"):
            effective_date = amendment.date
        
        if not effective_date:
            return score  # No date to validate
        
        # Try to parse date
        date_formats = [
            "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d",
            "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y"
        ]
        
        parsed_date = None
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(str(effective_date), fmt)
                break
            except ValueError:
                continue
        
        if not parsed_date:
            # Couldn't parse date
            return score - 0.2
        
        # Check if date is reasonable
        current_year = datetime.now().year
        date_year = parsed_date.year
        
        if 1800 <= date_year <= current_year + 1:  # Allow 1 year future
            score += 0.3
        else:
            score -= 0.3
        
        # Check date format consistency
        # If date includes day and month, it's more specific
        date_str = str(effective_date)
        if re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', date_str):
            score += 0.1  # Specific date format
        elif re.search(r'\b\d{4}\b', date_str):
            score += 0.05  # Just year
        
        return max(score, 0.1)
    
    def _extract_fields(self, amendment: Any) -> Dict[str, Any]:
        """Extract fields from amendment object.
        
        Args:
            amendment: Amendment object.
            
        Returns:
            Dictionary of extracted fields.
        """
        fields = {}
        
        # Common field patterns
        field_patterns = [
            ("section_number", ["section_number", "target_section", "section"]),
            ("act_number", ["act_number", "amendment_act_id", "target_act_number"]),
            ("act_year", ["act_year", "amendment_year", "target_year"]),
            ("amendment_type", ["amendment_type", "operation"]),
            ("page_num", ["page_num", "source_page", "page"]),
            ("effective_date", ["effective_date", "date", "notification_date"]),
            ("original_text", ["original_text", "text", "content"]),
            ("gazette_number", ["gazette_number", "gazette_ref"]),
        ]
        
        for target_field, source_attrs in field_patterns:
            for source_attr in source_attrs:
                if hasattr(amendment, source_attr):
                    value = getattr(amendment, source_attr)
                    if value is not None:
                        fields[target_field] = value
                        break
        
        return fields
    
    def _get_amendment_id(self, amendment: Any) -> str:
        """Get unique identifier for amendment.
        
        Args:
            amendment: Amendment object.
            
        Returns:
            Amendment identifier.
        """
        fields = self._extract_fields(amendment)
        
        # Create ID from key fields
        id_parts = []
        
        for field in ["section_number", "act_number", "act_year", "page_num"]:
            if field in fields:
                id_parts.append(str(fields[field]))
        
        return "_".join(id_parts) if id_parts else str(id(amendment))
    
    def _calculate_weighted_average(self, scores: Dict[str, float]) -> float:
        """Calculate weighted average of scores.
        
        Args:
            scores: Dictionary of factor scores.
            
        Returns:
            Weighted average score.
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor, score in scores.items():
            weight = self.weights.get(factor, 0.0)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return sum(scores.values()) / len(scores) if scores else 0.5
    
    def _determine_confidence_level(self, score: float) -> ConfidenceLevel:
        """Determine confidence level from score.
        
        Args:
            score: Confidence score.
            
        Returns:
            ConfidenceLevel enum.
        """
        if score >= ConfidenceLevel.VERY_HIGH.value:
            return ConfidenceLevel.VERY_HIGH
        elif score >= ConfidenceLevel.HIGH.value:
            return ConfidenceLevel.HIGH
        elif score >= ConfidenceLevel.MEDIUM.value:
            return ConfidenceLevel.MEDIUM
        elif score >= ConfidenceLevel.LOW.value:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


def create_confidence_scorer(weights: Optional[Dict[str, float]] = None) -> ConfidenceScorer:
    """Create a confidence scorer.
    
    Args:
        weights: Optional custom weights.
        
    Returns:
        ConfidenceScorer instance.
    """
    return ConfidenceScorer(weights=weights)


def score_amendment_confidence(
    amendment: Any,
    pdf_path: str,
    total_pages: int = 0,
    extraction_method: str = "",
    gazette_references: Optional[List[Dict[str, Any]]] = None
) -> ConfidenceScores:
    """Convenience function to score amendment confidence.
    
    Args:
        amendment: Amendment object.
        pdf_path: Path to PDF file.
        total_pages: Total pages in PDF.
        extraction_method: Extraction method used.
        gazette_references: Optional gazette references.
        
    Returns:
        ConfidenceScores object.
    """
    context = ExtractionContext(
        pdf_path=pdf_path,
        total_pages=total_pages,
        extraction_method=extraction_method,
        gazette_references=gazette_references or []
    )
    
    scorer = ConfidenceScorer()
    return scorer.calculate_confidence(amendment, context)