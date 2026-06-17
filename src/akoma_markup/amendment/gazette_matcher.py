"""Gazette-amendment matching for linking amendments to their gazette notifications.

Provides intelligent matching algorithms to connect extracted amendments
with their corresponding gazette publications.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from difflib import SequenceMatcher

from .visual_gazette import GazetteAnalysis

logger = logging.getLogger(__name__)


@dataclass
class GazetteMatch:
    """Represents a match between an amendment and a gazette."""
    
    amendment: Any  # Amendment object
    gazette: GazetteAnalysis
    match_score: float
    match_reasons: List[str]
    confidence: str = "medium"  # high/medium/low
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "amendment_id": self._get_amendment_id(),
            "gazette_path": str(self.gazette.pdf_path),
            "match_score": self.match_score,
            "match_reasons": self.match_reasons,
            "confidence": self.confidence,
            "amendment_details": self._get_amendment_details(),
            "gazette_details": self.gazette.metadata
        }
    
    def _get_amendment_id(self) -> str:
        """Get unique identifier for amendment."""
        try:
            # Try various attribute patterns
            attrs = ["section_number", "target_section", "section", "amendment_id"]
            for attr in attrs:
                if hasattr(self.amendment, attr):
                    value = getattr(self.amendment, attr)
                    if value:
                        return str(value)
            
            # Fallback to string representation
            return str(self.amendment)[:100]
        except:
            return "unknown"
    
    def _get_amendment_details(self) -> Dict[str, Any]:
        """Get amendment details for serialization."""
        details = {}
        
        # Try to extract common attributes
        common_attrs = [
            "section_number", "target_section", "section", "amendment_type",
            "act_number", "act_year", "effective_date", "page_num",
            "original_text", "amendment_act", "target_act"
        ]
        
        for attr in common_attrs:
            if hasattr(self.amendment, attr):
                value = getattr(self.amendment, attr)
                if value:
                    details[attr] = value
        
        return details


@dataclass
class MatchingResults:
    """Results of gazette-amendment matching."""
    
    matches: List[GazetteMatch]
    unmatched_amendments: List[Any]
    unmatched_gazettes: List[GazetteAnalysis]
    overall_match_rate: float
    matching_stats: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_matches": len(self.matches),
            "total_amendments": len(self.matches) + len(self.unmatched_amendments),
            "total_gazettes": len(self.matches) + len(self.unmatched_gazettes),
            "overall_match_rate": self.overall_match_rate,
            "matches": [m.to_dict() for m in self.matches],
            "unmatched_amendment_count": len(self.unmatched_amendments),
            "unmatched_gazette_count": len(self.unmatched_gazettes),
            "matching_stats": self.matching_stats or {},
            "summary": self._create_summary()
        }
    
    def _create_summary(self) -> Dict[str, Any]:
        """Create summary statistics."""
        match_scores = [m.match_score for m in self.matches]
        confidences = [m.confidence for m in self.matches]
        
        return {
            "average_match_score": sum(match_scores) / len(match_scores) if match_scores else 0,
            "high_confidence_matches": sum(1 for c in confidences if c == "high"),
            "medium_confidence_matches": sum(1 for c in confidences if c == "medium"),
            "low_confidence_matches": sum(1 for c in confidences if c == "low"),
        }


class GazetteAmendmentMatcher:
    """Matcher for connecting amendments to gazette notifications."""
    
    MIN_MATCH_SCORE = 0.6  # 60% minimum match score
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    MEDIUM_CONFIDENCE_THRESHOLD = 0.7
    
    def __init__(self, match_weights: Optional[Dict[str, float]] = None):
        """Initialize gazette-amendment matcher.
        
        Args:
            match_weights: Optional custom weights for matching factors.
        """
        self.match_weights = match_weights or {
            "act_year": 0.4,       # Year match is most important
            "act_number": 0.2,     # Act number match
            "date_proximity": 0.3,  # Date similarity
            "content_similarity": 0.1  # Text content match
        }
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.match_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Weights sum to {weight_sum}, normalizing to 1.0")
            for key in self.match_weights:
                self.match_weights[key] /= weight_sum
    
    def match_amendments_to_gazettes(
        self, 
        amendments: List[Any],
        gazettes: List[GazetteAnalysis]
    ) -> MatchingResults:
        """Match amendments to their corresponding gazette notifications.
        
        Args:
            amendments: List of amendment objects.
            gazettes: List of GazetteAnalysis objects.
            
        Returns:
            MatchingResults with matches and statistics.
        """
        logger.info(f"Matching {len(amendments)} amendments to {len(gazettes)} gazettes")
        
        matches = []
        unmatched_amendments = []
        unmatched_gazettes = list(gazettes)
        
        matching_stats = {
            "total_comparisons": 0,
            "matches_by_confidence": {"high": 0, "medium": 0, "low": 0},
            "match_scores": []
        }
        
        for amendment in amendments:
            best_match = None
            best_score = 0.0
            best_gazette = None
            
            for gazette in gazettes:
                matching_stats["total_comparisons"] += 1
                
                # Calculate matching score
                score = self._calculate_match_score(amendment, gazette)
                
                if score > best_score and score >= self.MIN_MATCH_SCORE:
                    best_score = score
                    best_match = gazette
            
            if best_match:
                # Determine confidence level
                confidence = self._determine_confidence(best_score)
                
                # Get match reasons
                match_reasons = self._get_match_reasons(amendment, best_match, best_score)
                
                # Create match object
                match_obj = GazetteMatch(
                    amendment=amendment,
                    gazette=best_match,
                    match_score=best_score,
                    match_reasons=match_reasons,
                    confidence=confidence
                )
                
                matches.append(match_obj)
                unmatched_gazettes.remove(best_match)
                
                # Update statistics
                matching_stats["matches_by_confidence"][confidence] += 1
                matching_stats["match_scores"].append(best_score)
                
                logger.debug(f"Matched amendment to {best_match.pdf_path.name} "
                           f"(score: {best_score:.2f}, confidence: {confidence})")
            else:
                unmatched_amendments.append(amendment)
                logger.debug(f"No match found for amendment")
        
        # Calculate overall match rate
        overall_match_rate = len(matches) / len(amendments) if amendments else 0
        
        # Update matching statistics
        matching_stats["average_match_score"] = (
            sum(matching_stats["match_scores"]) / len(matching_stats["match_scores"]) 
            if matching_stats["match_scores"] else 0
        )
        
        logger.info(f"Matching complete: {len(matches)} matches, "
                   f"{len(unmatched_amendments)} unmatched amendments, "
                   f"match rate: {overall_match_rate:.1%}")
        
        return MatchingResults(
            matches=matches,
            unmatched_amendments=unmatched_amendments,
            unmatched_gazettes=unmatched_gazettes,
            overall_match_rate=overall_match_rate,
            matching_stats=matching_stats
        )
    
    def _calculate_match_score(self, amendment: Any, gazette: GazetteAnalysis) -> float:
        """Calculate how well amendment matches gazette.
        
        Args:
            amendment: Amendment object.
            gazette: GazetteAnalysis object.
            
        Returns:
            Match score between 0 and 1.
        """
        score = 0.0
        
        # Extract amendment fields
        amendment_fields = self._extract_amendment_fields(amendment)
        gazette_metadata = gazette.metadata
        
        # Act/year match (highest weight)
        if (amendment_fields.get("act_year") and 
            gazette_metadata.get("act_year") and
            str(amendment_fields["act_year"]) == str(gazette_metadata["act_year"])):
            score += self.match_weights["act_year"] * 1.0
        else:
            # Partial year match (within 1 year)
            if (amendment_fields.get("act_year") and 
                gazette_metadata.get("act_year")):
                try:
                    amd_year = int(amendment_fields["act_year"])
                    gaz_year = int(gazette_metadata["act_year"])
                    if abs(amd_year - gaz_year) <= 1:
                        score += self.match_weights["act_year"] * 0.5
                except (ValueError, TypeError):
                    pass
        
        # Date proximity match
        date_score = self._calculate_date_match_score(
            amendment_fields.get("effective_date"),
            gazette_metadata.get("publication_date")
        )
        score += self.match_weights["date_proximity"] * date_score
        
        # Act number match
        if (amendment_fields.get("act_number") and 
            gazette_metadata.get("act_number") and
            str(amendment_fields["act_number"]) == str(gazette_metadata["act_number"])):
            score += self.match_weights["act_number"] * 1.0
        else:
            # Try to extract act number from various patterns
            amd_act_num = self._extract_act_number(amendment_fields.get("act_number", ""))
            gaz_act_num = self._extract_act_number(gazette_metadata.get("act_number", ""))
            
            if amd_act_num and gaz_act_num and amd_act_num == gaz_act_num:
                score += self.match_weights["act_number"] * 1.0
        
        # Content similarity match (text-based)
        content_score = self._calculate_content_match_score(amendment, gazette)
        score += self.match_weights["content_similarity"] * content_score
        
        # Additional boost for visual features match
        if gazette.visual_features and gazette.visual_features.get("is_official_looking"):
            # Official-looking gazettes get a small boost
            score *= 1.05
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _extract_amendment_fields(self, amendment: Any) -> Dict[str, Any]:
        """Extract fields from amendment object.
        
        Args:
            amendment: Amendment object.
            
        Returns:
            Dictionary with extracted fields.
        """
        fields = {}
        
        # Try to extract common attributes
        attr_mapping = [
            ("act_number", ["act_number", "amendment_act_id", "target_act_number"]),
            ("act_year", ["act_year", "amendment_year", "target_year"]),
            ("section_number", ["section_number", "target_section", "section"]),
            ("amendment_type", ["amendment_type", "operation"]),
            ("effective_date", ["effective_date", "date", "notification_date"]),
            ("original_text", ["original_text", "text", "content"]),
            ("page_num", ["page_num", "source_page", "page"]),
        ]
        
        for target_field, source_attrs in attr_mapping:
            for source_attr in source_attrs:
                if hasattr(amendment, source_attr):
                    value = getattr(amendment, source_attr)
                    if value:
                        fields[target_field] = value
                        break
        
        return fields
    
    def _calculate_date_match_score(self, amd_date: Optional[str], gaz_date: Optional[str]) -> float:
        """Calculate date match score.
        
        Args:
            amd_date: Amendment effective date.
            gaz_date: Gazette publication date.
            
        Returns:
            Date match score between 0 and 1.
        """
        if not amd_date or not gaz_date:
            return 0.0
        
        try:
            # Try various date formats
            date_formats = [
                "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d",
                "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y"
            ]
            
            amd_dt = None
            gaz_dt = None
            
            for fmt in date_formats:
                try:
                    if not amd_dt:
                        amd_dt = datetime.strptime(str(amd_date), fmt)
                    if not gaz_dt:
                        gaz_dt = datetime.strptime(str(gaz_date), fmt)
                except ValueError:
                    continue
            
            if not amd_dt or not gaz_dt:
                # Try to extract just year
                amd_year = re.search(r'\b(\d{4})\b', str(amd_date))
                gaz_year = re.search(r'\b(\d{4})\b', str(gaz_date))
                
                if amd_year and gaz_year and amd_year.group(1) == gaz_year.group(1):
                    return 0.8  # Same year
                elif amd_year and gaz_year and abs(int(amd_year.group(1)) - int(gaz_year.group(1))) <= 1:
                    return 0.5  # Within 1 year
                else:
                    return 0.2  # Different years
            
            # Calculate days difference
            days_diff = abs((amd_dt - gaz_dt).days)
            
            if days_diff == 0:
                return 1.0  # Exact match
            elif days_diff <= 30:
                return 0.8  # Within 30 days
            elif days_diff <= 90:
                return 0.5  # Within 90 days
            elif days_diff <= 365:
                return 0.3  # Within 1 year
            else:
                return 0.1  # More than 1 year
            
        except Exception as e:
            logger.debug(f"Date parsing failed: {e}")
            return 0.0
    
    def _extract_act_number(self, act_ref: str) -> Optional[str]:
        """Extract act number from reference string.
        
        Args:
            act_ref: Act reference string.
            
        Returns:
            Extracted act number, or None.
        """
        if not act_ref:
            return None
        
        # Look for Act X pattern
        match = re.search(r'Act\s+(\d+)', act_ref, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Look for standalone number
        match = re.search(r'\b(\d{1,4})\b', act_ref)
        if match:
            return match.group(1)
        
        return None
    
    def _calculate_content_match_score(self, amendment: Any, gazette: GazetteAnalysis) -> float:
        """Calculate content similarity score.
        
        Args:
            amendment: Amendment object.
            gazette: GazetteAnalysis object.
            
        Returns:
            Content match score between 0 and 1.
        """
        # Extract amendment text
        amd_text = ""
        if hasattr(amendment, "original_text"):
            amd_text = str(amendment.original_text)
        elif hasattr(amendment, "text"):
            amd_text = str(amendment.text)
        
        if not amd_text:
            return 0.0
        
        # Get gazette text (from OCR results or metadata)
        gaz_text = ""
        if gazette.ocr_results:
            # Combine all OCR text
            gaz_text = " ".join(gazette.ocr_results.values())
        elif "act_name" in gazette.metadata:
            gaz_text = gazette.metadata.get("act_name", "")
        
        if not gaz_text:
            return 0.0
        
        # Calculate text similarity
        similarity = SequenceMatcher(None, amd_text.lower(), gaz_text.lower()).ratio()
        
        # Look for keyword matches
        keywords = ["amendment", "section", "act", "insert", "delete", "substitute", "replace"]
        keyword_matches = sum(1 for kw in keywords if kw.lower() in amd_text.lower() and kw.lower() in gaz_text.lower())
        
        # Combined score
        return min(similarity + (keyword_matches * 0.1), 1.0)
    
    def _determine_confidence(self, match_score: float) -> str:
        """Determine confidence level based on match score.
        
        Args:
            match_score: Match score between 0 and 1.
            
        Returns:
            Confidence level: "high", "medium", or "low".
        """
        if match_score >= self.HIGH_CONFIDENCE_THRESHOLD:
            return "high"
        elif match_score >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return "medium"
        else:
            return "low"
    
    def _get_match_reasons(self, amendment: Any, gazette: GazetteAnalysis, match_score: float) -> List[str]:
        """Get reasons for match.
        
        Args:
            amendment: Amendment object.
            gazette: GazetteAnalysis object.
            match_score: Match score.
            
        Returns:
            List of match reason strings.
        """
        reasons = []
        
        # Extract fields
        amd_fields = self._extract_amendment_fields(amendment)
        gaz_metadata = gazette.metadata
        
        # Check year match
        if (amd_fields.get("act_year") and 
            gaz_metadata.get("act_year") and
            str(amd_fields["act_year"]) == str(gaz_metadata["act_year"])):
            reasons.append("Same act year")
        elif amd_fields.get("act_year") and gaz_metadata.get("act_year"):
            try:
                amd_year = int(amd_fields["act_year"])
                gaz_year = int(gaz_metadata["act_year"])
                if abs(amd_year - gaz_year) <= 1:
                    reasons.append("Act year within 1 year")
            except (ValueError, TypeError):
                pass
        
        # Check act number match
        if (amd_fields.get("act_number") and 
            gaz_metadata.get("act_number") and
            str(amd_fields["act_number"]) == str(gaz_metadata["act_number"])):
            reasons.append("Same act number")
        
        # Check date match
        date_score = self._calculate_date_match_score(
            amd_fields.get("effective_date"),
            gaz_metadata.get("publication_date")
        )
        if date_score > 0.7:
            reasons.append("Date closely matches")
        elif date_score > 0.3:
            reasons.append("Date somewhat matches")
        
        # Check visual features
        if gazette.visual_features and gazette.visual_features.get("is_official_looking"):
            reasons.append("Official-looking gazette")
        
        # Add match score
        reasons.append(f"Match score: {match_score:.2f}")
        
        return reasons


async def match_amendments_to_gazette_directory(
    amendments: List[Any],
    gazette_dir: Path,
    output_file: Optional[Path] = None
) -> MatchingResults:
    """Match amendments to gazettes in a directory.
    
    Args:
        amendments: List of amendment objects.
        gazette_dir: Directory containing gazette PDFs.
        output_file: Optional file to save matching results.
        
    Returns:
        MatchingResults with matches.
    """
    from .visual_gazette import analyze_gazette_directory
    
    logger.info(f"Matching amendments to gazettes in {gazette_dir}")
    
    # Analyze gazettes in directory
    analyses = await analyze_gazette_directory(gazette_dir)
    
    # Convert analyses to GazetteAnalysis objects
    gazettes = []
    for analysis_data in analyses.values():
        gazette = GazetteAnalysis.from_dict(analysis_data)
        gazettes.append(gazette)
    
    # Create matcher and match
    matcher = GazetteAmendmentMatcher()
    results = matcher.match_amendments_to_gazettes(amendments, gazettes)
    
    # Save results if output file specified
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved matching results to {output_file}")
    
    return results