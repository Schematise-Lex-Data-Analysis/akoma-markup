"""Tests for confidence module."""

import unittest
from unittest.mock import Mock
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.akoma_markup.amendment.confidence import (
    ConfidenceLevel,
    ConfidenceScores,
    ExtractionContext,
    ConfidenceScorer
)


class TestConfidenceLevel(unittest.TestCase):
    """Test ConfidenceLevel enum."""
    
    def test_confidence_level_values(self):
        """Test ConfidenceLevel enum values."""
        self.assertEqual(ConfidenceLevel.VERY_HIGH.value, 0.9)
        self.assertEqual(ConfidenceLevel.HIGH.value, 0.8)
        self.assertEqual(ConfidenceLevel.MEDIUM.value, 0.6)
        self.assertEqual(ConfidenceLevel.LOW.value, 0.4)
        self.assertEqual(ConfidenceLevel.VERY_LOW.value, 0.2)
    
    def test_confidence_level_from_value(self):
        """Test creating ConfidenceLevel from value."""
        self.assertEqual(ConfidenceLevel(0.9), ConfidenceLevel.VERY_HIGH)
        self.assertEqual(ConfidenceLevel(0.8), ConfidenceLevel.HIGH)
        self.assertEqual(ConfidenceLevel(0.6), ConfidenceLevel.MEDIUM)
        self.assertEqual(ConfidenceLevel(0.4), ConfidenceLevel.LOW)
        self.assertEqual(ConfidenceLevel(0.2), ConfidenceLevel.VERY_LOW)


class TestConfidenceScores(unittest.TestCase):
    """Test ConfidenceScores class."""
    
    def test_confidence_scores_init(self):
        """Test ConfidenceScores initialization."""
        factors = {"source": 0.9, "consistency": 0.8}
        weights = {"source": 0.5, "consistency": 0.5}
        
        scores = ConfidenceScores(
            overall=0.85,
            factors=factors,
            weights=weights,
            level=ConfidenceLevel.HIGH,
            recommendations=["Check section reference"]
        )
        
        self.assertEqual(scores.overall, 0.85)
        self.assertEqual(scores.factors, factors)
        self.assertEqual(scores.weights, weights)
        self.assertEqual(scores.level, ConfidenceLevel.HIGH)
        self.assertEqual(scores.recommendations, ["Check section reference"])
    
    def test_is_high_confidence_true(self):
        """Test is_high_confidence property when true."""
        scores = ConfidenceScores(
            overall=0.85,  # Above HIGH threshold (0.8)
            factors={},
            weights={},
            level=ConfidenceLevel.HIGH
        )
        
        self.assertTrue(scores.is_high_confidence)
    
    def test_is_high_confidence_false(self):
        """Test is_high_confidence property when false."""
        scores = ConfidenceScores(
            overall=0.7,  # Below HIGH threshold (0.8)
            factors={},
            weights={},
            level=ConfidenceLevel.MEDIUM
        )
        
        self.assertFalse(scores.is_high_confidence)
    
    def test_needs_review_true(self):
        """Test needs_review property when true."""
        scores = ConfidenceScores(
            overall=0.5,  # Below MEDIUM threshold (0.6)
            factors={},
            weights={},
            level=ConfidenceLevel.LOW
        )
        
        self.assertTrue(scores.needs_review)
    
    def test_needs_review_false(self):
        """Test needs_review property when false."""
        scores = ConfidenceScores(
            overall=0.7,  # Above MEDIUM threshold (0.6)
            factors={},
            weights={},
            level=ConfidenceLevel.HIGH
        )
        
        self.assertFalse(scores.needs_review)
    
    def test_to_dict(self):
        """Test ConfidenceScores to_dict method."""
        factors = {"source": 0.9, "consistency": 0.8}
        weights = {"source": 0.5, "consistency": 0.5}
        
        scores = ConfidenceScores(
            overall=0.85,
            factors=factors,
            weights=weights,
            level=ConfidenceLevel.HIGH,
            recommendations=["Verify with gazette"]
        )
        
        result = scores.to_dict()
        
        self.assertEqual(result["overall"], 0.85)
        self.assertEqual(result["level"], "HIGH")
        self.assertEqual(result["factors"], factors)
        self.assertEqual(result["weights"], weights)
        self.assertEqual(result["needs_review"], False)
        self.assertEqual(result["recommendations"], ["Verify with gazette"])


class TestExtractionContext(unittest.TestCase):
    """Test ExtractionContext class."""
    
    def test_extraction_context_init(self):
        """Test ExtractionContext initialization."""
        gazette_refs = [
            {"gazette_number": "123", "publication_date": "2023-01-15"},
            {"gazette_number": "456", "publication_date": "2023-02-20"}
        ]
        
        validation_results = {"pattern_match": 0.9, "section_ref": 0.8}
        
        context = ExtractionContext(
            pdf_path="/test/document.pdf",
            total_pages=50,
            extraction_method="hybrid",
            gazette_references=gazette_refs,
            validation_results=validation_results
        )
        
        self.assertEqual(context.pdf_path, "/test/document.pdf")
        self.assertEqual(context.total_pages, 50)
        self.assertEqual(context.extraction_method, "hybrid")
        self.assertEqual(context.gazette_references, gazette_refs)
        self.assertEqual(context.validation_results, validation_results)
    
    def test_has_gazette_context_true(self):
        """Test has_gazette_context property when true."""
        context = ExtractionContext(
            pdf_path="/test/document.pdf",
            gazette_references=[{"gazette_number": "123"}]
        )
        
        self.assertTrue(context.has_gazette_context)
    
    def test_has_gazette_context_false(self):
        """Test has_gazette_context property when false."""
        context = ExtractionContext(
            pdf_path="/test/document.pdf",
            gazette_references=[]
        )
        
        self.assertFalse(context.has_gazette_context)


class TestConfidenceScorer(unittest.TestCase):
    """Test ConfidenceScorer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scorer = ConfidenceScorer()
    
    def test_init_default(self):
        """Test default initialization."""
        self.assertIn("source", self.scorer.weights)
        self.assertIn("internal_consistency", self.scorer.weights)
        self.assertIn("cross_modal", self.scorer.weights)
        
        # Check weights sum to approximately 1.0
        weight_sum = sum(self.scorer.weights.values())
        self.assertAlmostEqual(weight_sum, 1.0, places=2)
        
        # Check source confidence mapping
        self.assertIn("vision_only", self.scorer.SOURCE_CONFIDENCE)
        self.assertIn("regex_only", self.scorer.SOURCE_CONFIDENCE)
        self.assertIn("vision+regex", self.scorer.SOURCE_CONFIDENCE)
    
    def test_init_custom_weights(self):
        """Test initialization with custom weights."""
        custom_weights = {
            "source": 0.3,
            "internal_consistency": 0.3,
            "cross_modal": 0.2,
            "pattern_match": 0.1,
            "contextual": 0.1,
        }
        
        scorer = ConfidenceScorer(weights=custom_weights)
        
        self.assertEqual(scorer.weights["source"], 0.3)
        self.assertEqual(scorer.weights["internal_consistency"], 0.3)
        self.assertEqual(scorer.weights["cross_modal"], 0.2)
    
    def test_init_normalizes_weights(self):
        """Test weights normalization when sum != 1.0."""
        unnormalized_weights = {
            "source": 0.4,
            "internal_consistency": 0.4,  # Sum = 0.8
        }
        
        scorer = ConfidenceScorer(weights=unnormalized_weights)
        
        # Should be normalized
        weight_sum = sum(scorer.weights.values())
        self.assertAlmostEqual(weight_sum, 1.0, places=2)
    
    def test_calculate_source_confidence_vision_only(self):
        """Test source confidence calculation for vision-only extraction."""
        amendment = Mock()
        amendment.extraction_method = "vision"
        amendment.has_regex_match = False
        
        confidence = self.scorer._calculate_source_confidence(amendment)
        
        self.assertAlmostEqual(confidence, self.scorer.SOURCE_CONFIDENCE["vision_only"], places=2)
    
    def test_calculate_source_confidence_regex_only(self):
        """Test source confidence calculation for regex-only extraction."""
        amendment = Mock()
        amendment.extraction_method = "regex"
        amendment.has_vision_match = False
        
        confidence = self.scorer._calculate_source_confidence(amendment)
        
        self.assertAlmostEqual(confidence, self.scorer.SOURCE_CONFIDENCE["regex_only"], places=2)
    
    def test_calculate_source_confidence_both(self):
        """Test source confidence calculation when both methods agree."""
        amendment = Mock()
        amendment.extraction_method = "hybrid"
        amendment.has_vision_match = True
        amendment.has_regex_match = True
        amendment.vision_regex_agreement = True
        
        confidence = self.scorer._calculate_source_confidence(amendment)
        
        self.assertAlmostEqual(confidence, self.scorer.SOURCE_CONFIDENCE["vision+regex"], places=2)
    
    def test_calculate_internal_consistency(self):
        """Test internal consistency calculation."""
        amendment = Mock()
        amendment.section_number = "45A"
        amendment.operation = "insert"
        amendment.text_content = "Insert new section 45A"
        
        # Mock consistency checks
        amendment.is_section_number_valid = True
        amendment.is_operation_valid = True
        amendment.has_consistent_formatting = True
        
        consistency = self.scorer._calculate_internal_consistency(amendment)
        
        self.assertGreaterEqual(consistency, 0.0)
        self.assertLessEqual(consistency, 1.0)
    
    def test_calculate_cross_modal_agreement(self):
        """Test cross-modal agreement calculation."""
        amendment = Mock()
        amendment.has_vision_match = True
        amendment.has_regex_match = True
        amendment.vision_regex_similarity = 0.9
        
        agreement = self.scorer._calculate_cross_modal_agreement(amendment)
        
        self.assertGreaterEqual(agreement, 0.0)
        self.assertLessEqual(agreement, 1.0)
    
    def test_calculate_pattern_match(self):
        """Test pattern match quality calculation."""
        amendment = Mock()
        amendment.matches_expected_pattern = True
        amendment.pattern_confidence = 0.8
        
        pattern_score = self.scorer._calculate_pattern_match(amendment)
        
        self.assertGreaterEqual(pattern_score, 0.0)
        self.assertLessEqual(pattern_score, 1.0)
    
    def test_calculate_contextual_validation(self):
        """Test contextual validation calculation."""
        amendment = Mock()
        amendment.is_within_document_context = True
        amendment.fits_timeline = True
        
        context = ExtractionContext(pdf_path="/test/document.pdf")
        
        contextual = self.scorer._calculate_contextual_validation(amendment, context)
        
        self.assertGreaterEqual(contextual, 0.0)
        self.assertLessEqual(contextual, 1.0)
    
    def test_calculate_gazette_reference(self):
        """Test gazette reference calculation."""
        amendment = Mock()
        amendment.gazette_references = [{"match_score": 0.9}, {"match_score": 0.8}]
        
        context = ExtractionContext(
            pdf_path="/test/document.pdf",
            gazette_references=[{"gazette_number": "123"}]
        )
        
        gazette_score = self.scorer._calculate_gazette_reference(amendment, context)
        
        self.assertGreaterEqual(gazette_score, 0.0)
        self.assertLessEqual(gazette_score, 1.0)
    
    def test_calculate_temporal_consistency(self):
        """Test temporal consistency calculation."""
        amendment = Mock()
        amendment.effective_date = "2023-01-15"
        amendment.gazette_date = "2023-01-10"
        
        temporal = self.scorer._calculate_temporal_consistency(amendment)
        
        self.assertGreaterEqual(temporal, 0.0)
        self.assertLessEqual(temporal, 1.0)
    
    def test_determine_confidence_level(self):
        """Test confidence level determination."""
        test_cases = [
            (0.95, ConfidenceLevel.VERY_HIGH),
            (0.85, ConfidenceLevel.HIGH),
            (0.70, ConfidenceLevel.HIGH),
            (0.65, ConfidenceLevel.MEDIUM),
            (0.50, ConfidenceLevel.MEDIUM),
            (0.45, ConfidenceLevel.LOW),
            (0.30, ConfidenceLevel.LOW),
            (0.15, ConfidenceLevel.VERY_LOW),
            (0.0, ConfidenceLevel.VERY_LOW),
        ]
        
        for score, expected_level in test_cases:
            with self.subTest(score=score):
                level = self.scorer._determine_confidence_level(score)
                self.assertEqual(level, expected_level)


if __name__ == '__main__':
    unittest.main()