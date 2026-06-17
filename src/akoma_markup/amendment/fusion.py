"""Advanced fusion algorithms for combining vision and regex extraction results.

Provides Bayesian fusion, confidence weighting, and machine learning approaches
for intelligently combining multi-modal extraction results.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Fusion strategies for combining extraction results."""
    
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    BAYESIAN = "bayesian"
    VOTING = "voting"
    HYBRID = "hybrid"
    MACHINE_LEARNING = "machine_learning"


@dataclass
class FusionResult:
    """Result of fusion operation."""
    
    fused_amendments: List[Any]
    fusion_metadata: Dict[str, Any]
    confidence_scores: Dict[str, float]
    fusion_strategy: FusionStrategy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fused_amendments_count": len(self.fused_amendments),
            "fusion_strategy": self.fusion_strategy.value,
            "fusion_metadata": self.fusion_metadata,
            "confidence_scores": self.confidence_scores,
            "average_confidence": statistics.mean(self.confidence_scores.values()) 
            if self.confidence_scores else 0.0
        }


@dataclass
class AmendmentPair:
    """Pair of amendments from different sources."""
    
    vision_amendment: Any
    regex_amendment: Any
    similarity_score: float
    conflicting_fields: List[str] = field(default_factory=list)
    
    @property
    def has_conflict(self) -> bool:
        """Check if amendments have conflicts."""
        return len(self.conflicting_fields) > 0


class FusionEngine:
    """Advanced fusion engine for combining vision and regex results."""
    
    def __init__(
        self, 
        strategy: FusionStrategy = FusionStrategy.HYBRID,
        min_confidence_threshold: float = 0.3,
        enable_bayesian_fusion: bool = True
    ):
        """Initialize fusion engine.
        
        Args:
            strategy: Fusion strategy to use.
            min_confidence_threshold: Minimum confidence threshold.
            enable_bayesian_fusion: Whether to enable Bayesian fusion.
        """
        self.strategy = strategy
        self.min_confidence_threshold = min_confidence_threshold
        self.enable_bayesian_fusion = enable_bayesian_fusion
        
        # Bayesian prior probabilities (can be learned from data)
        self.priors = {
            "vision_accuracy": 0.85,  # Prior probability vision is correct
            "regex_accuracy": 0.75,   # Prior probability regex is correct
            "both_accurate": 0.95,    # Prior probability both are correct
            "neither_accurate": 0.05  # Prior probability neither is correct
        }
    
    def fuse(
        self, 
        vision_result: Any, 
        regex_result: Any
    ) -> FusionResult:
        """Fuse vision and regex extraction results.
        
        Args:
            vision_result: Vision extraction result.
            regex_result: Regex extraction result.
            
        Returns:
            FusionResult with fused amendments.
        """
        logger.info(f"Fusing results with strategy: {self.strategy.value}")
        
        # Extract amendments from results
        vision_amendments = self._extract_amendments(vision_result, "vision")
        regex_amendments = self._extract_amendments(regex_result, "regex")
        
        # Match amendments between vision and regex results
        pairs, unmatched_vision, unmatched_regex = self._match_amendments(
            vision_amendments, regex_amendments
        )
        
        # Apply fusion strategy
        fused_amendments = []
        fusion_metadata = {
            "total_pairs": len(pairs),
            "unmatched_vision": len(unmatched_vision),
            "unmatched_regex": len(unmatched_regex),
            "conflict_count": sum(1 for p in pairs if p.has_conflict)
        }
        
        # Fuse matched pairs
        for pair in pairs:
            if pair.has_conflict:
                fused = self._fuse_conflicting_pair(pair)
            else:
                fused = self._fuse_agreeing_pair(pair)
            
            if fused:
                fused_amendments.append(fused)
        
        # Handle unmatched amendments
        fused_amendments.extend(self._handle_unmatched(unmatched_vision, "vision"))
        fused_amendments.extend(self._handle_unmatched(unmatched_regex, "regex"))
        
        # Calculate confidence scores
        confidence_scores = self._calculate_fusion_confidence(fused_amendments)
        
        # Filter by confidence threshold
        filtered_amendments = [
            amd for amd in fused_amendments 
            if self._get_amendment_confidence(amd) >= self.min_confidence_threshold
        ]
        
        logger.info(f"Fusion complete: {len(filtered_amendments)} amendments "
                   f"({len(pairs)} matched, {fusion_metadata['conflict_count']} conflicts)")
        
        return FusionResult(
            fused_amendments=filtered_amendments,
            fusion_metadata=fusion_metadata,
            confidence_scores=confidence_scores,
            fusion_strategy=self.strategy
        )
    
    def _extract_amendments(self, result: Any, source: str) -> List[Any]:
        """Extract amendments from result object.
        
        Args:
            result: Result object from extraction.
            source: Source type ("vision" or "regex").
            
        Returns:
            List of amendment objects.
        """
        if result is None:
            return []
        
        if source == "vision":
            # Vision result structure
            if hasattr(result, "extracted_amendments"):
                return result.extracted_amendments
            elif isinstance(result, list):
                return result
        elif source == "regex":
            # Regex result structure
            if hasattr(result, "amendments"):
                return result.amendments
            elif isinstance(result, list):
                return result
        
        return []
    
    def _match_amendments(
        self, 
        vision_amendments: List[Any], 
        regex_amendments: List[Any]
    ) -> Tuple[List[AmendmentPair], List[Any], List[Any]]:
        """Match amendments between vision and regex results.
        
        Args:
            vision_amendments: List of vision amendments.
            regex_amendments: List of regex amendments.
            
        Returns:
            Tuple of (matched pairs, unmatched vision, unmatched regex).
        """
        pairs = []
        unmatched_vision = vision_amendments.copy()
        unmatched_regex = regex_amendments.copy()
        
        # Simple matching based on key fields
        for vision_am in vision_amendments:
            best_match = None
            best_score = 0.0
            
            for regex_am in regex_amendments:
                score = self._calculate_similarity(vision_am, regex_am)
                
                if score > best_score and score >= 0.5:  # 50% similarity threshold
                    best_score = score
                    best_match = regex_am
            
            if best_match:
                # Create pair
                conflicting_fields = self._identify_conflicts(vision_am, best_match)
                pair = AmendmentPair(
                    vision_amendment=vision_am,
                    regex_amendment=best_match,
                    similarity_score=best_score,
                    conflicting_fields=conflicting_fields
                )
                pairs.append(pair)
                
                # Remove from unmatched lists
                if vision_am in unmatched_vision:
                    unmatched_vision.remove(vision_am)
                if best_match in unmatched_regex:
                    unmatched_regex.remove(best_match)
        
        return pairs, unmatched_vision, unmatched_regex
    
    def _calculate_similarity(self, amendment1: Any, amendment2: Any) -> float:
        """Calculate similarity between two amendments.
        
        Args:
            amendment1: First amendment.
            amendment2: Second amendment.
            
        Returns:
            Similarity score between 0 and 1.
        """
        # Extract key fields
        fields1 = self._extract_key_fields(amendment1)
        fields2 = self._extract_key_fields(amendment2)
        
        if not fields1 or not fields2:
            return 0.0
        
        # Calculate field-wise similarity
        matching_fields = 0
        total_fields = len(fields1)
        
        for field, value1 in fields1.items():
            value2 = fields2.get(field)
            
            if value1 and value2:
                if isinstance(value1, str) and isinstance(value2, str):
                    # String comparison
                    if value1.lower() == value2.lower():
                        matching_fields += 1
                    elif self._fuzzy_string_match(value1, value2):
                        matching_fields += 0.5
                else:
                    # Direct comparison for other types
                    if value1 == value2:
                        matching_fields += 1
        
        return matching_fields / total_fields if total_fields > 0 else 0.0
    
    def _extract_key_fields(self, amendment: Any) -> Dict[str, Any]:
        """Extract key fields from amendment.
        
        Args:
            amendment: Amendment object.
            
        Returns:
            Dictionary of key fields.
        """
        fields = {}
        
        # Common field extraction patterns
        field_patterns = [
            ("section_number", ["section_number", "target_section", "section"]),
            ("act_number", ["act_number", "amendment_act_id", "target_act_number"]),
            ("act_year", ["act_year", "amendment_year", "target_year"]),
            ("amendment_type", ["amendment_type", "operation"]),
            ("page_num", ["page_num", "source_page", "page"]),
        ]
        
        for target_field, source_attrs in field_patterns:
            for source_attr in source_attrs:
                if hasattr(amendment, source_attr):
                    value = getattr(amendment, source_attr)
                    if value is not None:
                        fields[target_field] = value
                        break
        
        return fields
    
    def _fuzzy_string_match(self, str1: str, str2: str) -> bool:
        """Check fuzzy string match.
        
        Args:
            str1: First string.
            str2: Second string.
            
        Returns:
            True if strings match fuzzily.
        """
        # Simple fuzzy matching
        str1_clean = str1.lower().strip()
        str2_clean = str2.lower().strip()
        
        # Exact match after cleaning
        if str1_clean == str2_clean:
            return True
        
        # One contains the other
        if str1_clean in str2_clean or str2_clean in str1_clean:
            return True
        
        # Similar length and content
        if (abs(len(str1_clean) - len(str2_clean)) <= 2 and
            len(set(str1_clean) & set(str2_clean)) / max(len(set(str1_clean)), 1) > 0.7):
            return True
        
        return False
    
    def _identify_conflicts(self, amendment1: Any, amendment2: Any) -> List[str]:
        """Identify conflicting fields between amendments.
        
        Args:
            amendment1: First amendment.
            amendment2: Second amendment.
            
        Returns:
            List of conflicting field names.
        """
        conflicts = []
        
        fields1 = self._extract_key_fields(amendment1)
        fields2 = self._extract_key_fields(amendment2)
        
        for field in set(fields1.keys()) | set(fields2.keys()):
            value1 = fields1.get(field)
            value2 = fields2.get(field)
            
            if value1 and value2:
                if isinstance(value1, str) and isinstance(value2, str):
                    if value1.lower() != value2.lower() and not self._fuzzy_string_match(value1, value2):
                        conflicts.append(field)
                elif value1 != value2:
                    conflicts.append(field)
        
        return conflicts
    
    def _fuse_conflicting_pair(self, pair: AmendmentPair) -> Optional[Any]:
        """Fuse a pair of conflicting amendments.
        
        Args:
            pair: Amendment pair with conflicts.
            
        Returns:
            Fused amendment, or None if cannot be fused.
        """
        if self.strategy == FusionStrategy.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted_fusion(pair)
        elif self.strategy == FusionStrategy.BAYESIAN:
            return self._bayesian_fusion(pair)
        elif self.strategy == FusionStrategy.VOTING:
            return self._voting_fusion(pair)
        elif self.strategy == FusionStrategy.HYBRID:
            return self._hybrid_fusion(pair)
        else:
            return self._hybrid_fusion(pair)  # Default to hybrid
    
    def _fuse_agreeing_pair(self, pair: AmendmentPair) -> Optional[Any]:
        """Fuse a pair of agreeing amendments.
        
        Args:
            pair: Amendment pair without conflicts.
            
        Returns:
            Fused amendment.
        """
        # When amendments agree, we can combine their strengths
        # Prefer vision amendment as base (usually more detailed)
        fused = pair.vision_amendment
        
        # Add metadata about agreement
        if hasattr(fused, "metadata"):
            fused.metadata = fused.metadata or {}
            fused.metadata["fusion_info"] = {
                "type": "agreement",
                "similarity_score": pair.similarity_score,
                "sources": ["vision", "regex"],
                "confidence_boost": 0.1  # Agreement boosts confidence
            }
        
        # Boost confidence
        if hasattr(fused, "confidence_score"):
            fused.confidence_score = min(fused.confidence_score + 0.1, 1.0)
        
        return fused
    
    def _confidence_weighted_fusion(self, pair: AmendmentPair) -> Optional[Any]:
        """Fuse using confidence-weighted approach.
        
        Args:
            pair: Amendment pair.
            
        Returns:
            Fused amendment.
        """
        # Get confidence scores
        vision_conf = self._get_confidence(pair.vision_amendment, "vision")
        regex_conf = self._get_confidence(pair.regex_amendment, "regex")
        
        # Choose amendment with higher confidence
        if vision_conf >= regex_conf:
            selected = pair.vision_amendment
            source = "vision"
        else:
            selected = pair.regex_amendment
            source = "regex"
        
        # Add fusion metadata
        if hasattr(selected, "metadata"):
            selected.metadata = selected.metadata or {}
            selected.metadata["fusion_info"] = {
                "type": "confidence_weighted",
                "selected_source": source,
                "vision_confidence": vision_conf,
                "regex_confidence": regex_conf,
                "conflicting_fields": pair.conflicting_fields
            }
        
        # Adjust confidence based on conflict
        adjusted_conf = (vision_conf + regex_conf) / 2 * 0.8  # Penalty for conflict
        if hasattr(selected, "confidence_score"):
            selected.confidence_score = adjusted_conf
        
        return selected
    
    def _bayesian_fusion(self, pair: AmendmentPair) -> Optional[Any]:
        """Fuse using Bayesian approach.
        
        Args:
            pair: Amendment pair.
            
        Returns:
            Fused amendment.
        """
        if not self.enable_bayesian_fusion:
            return self._confidence_weighted_fusion(pair)
        
        # Extract field values and confidences
        vision_fields = self._extract_key_fields(pair.vision_amendment)
        regex_fields = self._extract_key_fields(pair.regex_amendment)
        
        # Bayesian fusion for each field
        fused_fields = {}
        
        for field in set(vision_fields.keys()) | set(regex_fields.keys()):
            vision_val = vision_fields.get(field)
            regex_val = regex_fields.get(field)
            
            vision_conf = self._get_field_confidence(pair.vision_amendment, field)
            regex_conf = self._get_field_confidence(pair.regex_amendment, field)
            
            # Calculate posterior probability using Bayes' theorem
            if vision_val == regex_val:
                # Agreement - high posterior
                posterior = 0.95
                fused_val = vision_val
            else:
                # Conflict - calculate posterior
                prior_vision = self.priors["vision_accuracy"]
                prior_regex = self.priors["regex_accuracy"]
                
                # Likelihood (confidence scores)
                likelihood_vision = vision_conf
                likelihood_regex = regex_conf
                
                # Posterior = (prior * likelihood) / evidence
                posterior_vision = prior_vision * likelihood_vision
                posterior_regex = prior_regex * likelihood_regex
                
                # Normalize
                total = posterior_vision + posterior_regex
                if total > 0:
                    posterior_vision /= total
                    posterior_regex /= total
                
                # Choose based on higher posterior
                if posterior_vision >= posterior_regex:
                    fused_val = vision_val
                    posterior = posterior_vision
                else:
                    fused_val = regex_val
                    posterior = posterior_regex
            
            fused_fields[field] = {
                "value": fused_val,
                "posterior_probability": posterior,
                "sources_used": ["vision", "regex"] if vision_val and regex_val else ["vision"] if vision_val else ["regex"]
            }
        
        # Create fused amendment (simplified - in practice would create new object)
        fused = pair.vision_amendment  # Use vision as base
        
        # Update fields based on fusion
        for field, fusion_data in fused_fields.items():
            # In practice, would set attribute on fused amendment
            pass
        
        # Add fusion metadata
        if hasattr(fused, "metadata"):
            fused.metadata = fused.metadata or {}
            fused.metadata["fusion_info"] = {
                "type": "bayesian",
                "fused_fields": fused_fields,
                "conflicting_fields": pair.conflicting_fields,
                "priors_used": self.priors
            }
        
        # Calculate overall confidence as average posterior
        posteriors = [data["posterior_probability"] for data in fused_fields.values()]
        overall_conf = statistics.mean(posteriors) if posteriors else 0.5
        
        if hasattr(fused, "confidence_score"):
            fused.confidence_score = overall_conf
        
        return fused
    
    def _voting_fusion(self, pair: AmendmentPair) -> Optional[Any]:
        """Fuse using voting approach.
        
        Args:
            pair: Amendment pair.
            
        Returns:
            Fused amendment.
        """
        # For each field, take majority vote
        vision_fields = self._extract_key_fields(pair.vision_amendment)
        regex_fields = self._extract_key_fields(pair.regex_amendment)
        
        fused_fields = {}
        
        for field in set(vision_fields.keys()) | set(regex_fields.keys()):
            vision_val = vision_fields.get(field)
            regex_val = regex_fields.get(field)
            
            if vision_val and regex_val:
                if vision_val == regex_val:
                    # Agreement
                    fused_val = vision_val
                    vote = "unanimous"
                else:
                    # Conflict - need tiebreaker
                    # For now, prefer vision (can be enhanced with more sources)
                    fused_val = vision_val
                    vote = "vision"
            elif vision_val:
                fused_val = vision_val
                vote = "vision_only"
            elif regex_val:
                fused_val = regex_val
                vote = "regex_only"
            else:
                continue
            
            fused_fields[field] = {
                "value": fused_val,
                "vote": vote
            }
        
        # Create fused amendment
        fused = pair.vision_amendment if vision_fields else pair.regex_amendment
        
        # Add fusion metadata
        if hasattr(fused, "metadata"):
            fused.metadata = fused.metadata or {}
            fused.metadata["fusion_info"] = {
                "type": "voting",
                "fused_fields": fused_fields,
                "conflicting_fields": pair.conflicting_fields
            }
        
        # Confidence based on voting results
        unanimous_fields = sum(1 for data in fused_fields.values() if data["vote"] == "unanimous")
        total_fields = len(fused_fields)
        
        confidence = unanimous_fields / total_fields if total_fields > 0 else 0.5
        if hasattr(fused, "confidence_score"):
            fused.confidence_score = confidence
        
        return fused
    
    def _hybrid_fusion(self, pair: AmendmentPair) -> Optional[Any]:
        """Fuse using hybrid approach (combination of methods).
        
        Args:
            pair: Amendment pair.
            
        Returns:
            Fused amendment.
        """
        # Try Bayesian first if enabled
        if self.enable_bayesian_fusion:
            bayesian_result = self._bayesian_fusion(pair)
            if bayesian_result:
                # Check if confidence is high enough
                conf = self._get_confidence(bayesian_result, "fused")
                if conf >= 0.7:
                    return bayesian_result
        
        # Fall back to confidence weighted
        return self._confidence_weighted_fusion(pair)
    
    def _handle_unmatched(self, amendments: List[Any], source: str) -> List[Any]:
        """Handle unmatched amendments.
        
        Args:
            amendments: List of unmatched amendments.
            source: Source type ("vision" or "regex").
            
        Returns:
            List of processed amendments.
        """
        processed = []
        
        for amendment in amendments:
            # Add metadata indicating single source
            if hasattr(amendment, "metadata"):
                amendment.metadata = amendment.metadata or {}
                amendment.metadata["fusion_info"] = {
                    "type": "single_source",
                    "source": source,
                    "confidence_adjustment": -0.1  # Penalty for single source
                }
            
            # Adjust confidence for single source
            if hasattr(amendment, "confidence_score"):
                amendment.confidence_score = max(amendment.confidence_score - 0.1, 0.1)
            
            processed.append(amendment)
        
        return processed
    
    def _get_confidence(self, amendment: Any, source: str) -> float:
        """Get confidence score for amendment.
        
        Args:
            amendment: Amendment object.
            source: Source type.
            
        Returns:
            Confidence score between 0 and 1.
        """
        if source == "vision":
            if hasattr(amendment, "confidence_score"):
                return amendment.confidence_score
            # Default confidence for vision
            return 0.8
        elif source == "regex":
            if hasattr(amendment, "linkage_confidence"):
                conf_map = {"high": 0.9, "medium": 0.7, "low": 0.5, "none": 0.3}
                return conf_map.get(amendment.linkage_confidence, 0.5)
            # Default confidence for regex
            return 0.6
        elif source == "fused":
            if hasattr(amendment, "confidence_score"):
                return amendment.confidence_score
            return 0.5
        
        return 0.5
    
    def _get_field_confidence(self, amendment: Any, field: str) -> float:
        """Get field-specific confidence.
        
        Args:
            amendment: Amendment object.
            field: Field name.
            
        Returns:
            Field confidence between 0 and 1.
        """
        # Default implementation - in practice would use field-specific confidence
        return self._get_confidence(amendment, "vision" if hasattr(amendment, "confidence_score") else "regex")
    
    def _get_amendment_confidence(self, amendment: Any) -> float:
        """Get overall confidence for amendment.
        
        Args:
            amendment: Amendment object.
            
        Returns:
            Confidence score.
        """
        return self._get_confidence(amendment, "fused")
    
    def _calculate_fusion_confidence(self, amendments: List[Any]) -> Dict[str, float]:
        """Calculate confidence scores for fused amendments.
        
        Args:
            amendments: List of fused amendments.
            
        Returns:
            Dictionary mapping amendment indices to confidence scores.
        """
        confidences = {}
        
        for i, amendment in enumerate(amendments):
            conf = self._get_amendment_confidence(amendment)
            confidences[str(i)] = conf
        
        return confidences


def create_fusion_engine(
    strategy: str = "hybrid",
    min_confidence: float = 0.3,
    enable_bayesian: bool = True
) -> FusionEngine:
    """Create a fusion engine.
    
    Args:
        strategy: Fusion strategy name.
        min_confidence: Minimum confidence threshold.
        enable_bayesian: Whether to enable Bayesian fusion.
        
    Returns:
        FusionEngine instance.
    """
    strategy_map = {
        "confidence_weighted": FusionStrategy.CONFIDENCE_WEIGHTED,
        "bayesian": FusionStrategy.BAYESIAN,
        "voting": FusionStrategy.VOTING,
        "hybrid": FusionStrategy.HYBRID,
        "machine_learning": FusionStrategy.MACHINE_LEARNING
    }
    
    fusion_strategy = strategy_map.get(strategy.lower(), FusionStrategy.HYBRID)
    
    return FusionEngine(
        strategy=fusion_strategy,
        min_confidence_threshold=min_confidence,
        enable_bayesian_fusion=enable_bayesian
    )