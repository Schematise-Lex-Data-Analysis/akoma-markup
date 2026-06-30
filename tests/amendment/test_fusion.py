"""Tests for fusion module."""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.akoma_markup.amendment.fusion import (
    FusionStrategy,
    FusionResult,
    AmendmentPair,
    FusionEngine
)


class TestFusionStrategy(unittest.TestCase):
    """Test FusionStrategy enum."""
    
    def test_fusion_strategy_values(self):
        """Test FusionStrategy enum values."""
        self.assertEqual(FusionStrategy.CONFIDENCE_WEIGHTED.value, "confidence_weighted")
        self.assertEqual(FusionStrategy.BAYESIAN.value, "bayesian")
        self.assertEqual(FusionStrategy.VOTING.value, "voting")
        self.assertEqual(FusionStrategy.HYBRID.value, "hybrid")
        self.assertEqual(FusionStrategy.MACHINE_LEARNING.value, "machine_learning")
    
    def test_fusion_strategy_from_string(self):
        """Test creating FusionStrategy from string."""
        self.assertEqual(FusionStrategy("confidence_weighted"), FusionStrategy.CONFIDENCE_WEIGHTED)
        self.assertEqual(FusionStrategy("bayesian"), FusionStrategy.BAYESIAN)
        self.assertEqual(FusionStrategy("hybrid"), FusionStrategy.HYBRID)


class TestFusionResult(unittest.TestCase):
    """Test FusionResult class."""
    
    def test_fusion_result_init(self):
        """Test FusionResult initialization."""
        fused_amendments = [Mock(), Mock()]
        fusion_metadata = {"total_pairs": 5, "conflict_count": 2}
        confidence_scores = {"amendment1": 0.8, "amendment2": 0.9}
        
        result = FusionResult(
            fused_amendments=fused_amendments,
            fusion_metadata=fusion_metadata,
            confidence_scores=confidence_scores,
            fusion_strategy=FusionStrategy.HYBRID
        )
        
        self.assertEqual(len(result.fused_amendments), 2)
        self.assertEqual(result.fusion_metadata, fusion_metadata)
        self.assertEqual(result.confidence_scores, confidence_scores)
        self.assertEqual(result.fusion_strategy, FusionStrategy.HYBRID)
    
    def test_fusion_result_to_dict(self):
        """Test FusionResult to_dict method."""
        fused_amendments = [Mock(), Mock(), Mock()]
        fusion_metadata = {"total_pairs": 5, "conflict_count": 1}
        confidence_scores = {"amendment1": 0.8, "amendment2": 0.9, "amendment3": 0.7}
        
        result = FusionResult(
            fused_amendments=fused_amendments,
            fusion_metadata=fusion_metadata,
            confidence_scores=confidence_scores,
            fusion_strategy=FusionStrategy.BAYESIAN
        )
        
        result_dict = result.to_dict()
        
        self.assertEqual(result_dict["fused_amendments_count"], 3)
        self.assertEqual(result_dict["fusion_strategy"], "bayesian")
        self.assertEqual(result_dict["fusion_metadata"], fusion_metadata)
        self.assertEqual(result_dict["confidence_scores"], confidence_scores)
        self.assertAlmostEqual(result_dict["average_confidence"], 0.8, places=2)
    
    def test_fusion_result_to_dict_empty(self):
        """Test FusionResult to_dict with empty confidence scores."""
        result = FusionResult(
            fused_amendments=[],
            fusion_metadata={},
            confidence_scores={},
            fusion_strategy=FusionStrategy.CONFIDENCE_WEIGHTED
        )
        
        result_dict = result.to_dict()
        self.assertEqual(result_dict["average_confidence"], 0.0)


class TestAmendmentPair(unittest.TestCase):
    """Test AmendmentPair class."""
    
    def test_amendment_pair_init(self):
        """Test AmendmentPair initialization."""
        vision_amendment = Mock()
        vision_amendment.section_number = "45A"
        vision_amendment.operation = "insert"
        
        regex_amendment = Mock()
        regex_amendment.section_number = "45A"
        regex_amendment.operation = "insert"
        
        pair = AmendmentPair(
            vision_amendment=vision_amendment,
            regex_amendment=regex_amendment,
            similarity_score=0.9,
            conflicting_fields=["text_content"]
        )
        
        self.assertEqual(pair.vision_amendment, vision_amendment)
        self.assertEqual(pair.regex_amendment, regex_amendment)
        self.assertEqual(pair.similarity_score, 0.9)
        self.assertEqual(pair.conflicting_fields, ["text_content"])
    
    def test_has_conflict_true(self):
        """Test has_conflict property when conflicts exist."""
        pair = AmendmentPair(
            vision_amendment=Mock(),
            regex_amendment=Mock(),
            similarity_score=0.8,
            conflicting_fields=["section_number", "operation"]
        )
        
        self.assertTrue(pair.has_conflict)
    
    def test_has_conflict_false(self):
        """Test has_conflict property when no conflicts."""
        pair = AmendmentPair(
            vision_amendment=Mock(),
            regex_amendment=Mock(),
            similarity_score=0.9,
            conflicting_fields=[]
        )
        
        self.assertFalse(pair.has_conflict)


class TestFusionEngine(unittest.TestCase):
    """Test FusionEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = FusionEngine(
            strategy=FusionStrategy.HYBRID,
            min_confidence_threshold=0.3,
            enable_bayesian_fusion=True
        )
    
    def test_init_default(self):
        """Test default initialization."""
        engine = FusionEngine()
        
        self.assertEqual(engine.strategy, FusionStrategy.HYBRID)
        self.assertEqual(engine.min_confidence_threshold, 0.3)
        self.assertTrue(engine.enable_bayesian_fusion)
        
        # Check priors
        self.assertIn("vision_accuracy", engine.priors)
        self.assertIn("regex_accuracy", engine.priors)
        self.assertIn("both_accurate", engine.priors)
        self.assertIn("neither_accurate", engine.priors)
    
    def test_init_custom(self):
        """Test custom initialization."""
        engine = FusionEngine(
            strategy=FusionStrategy.BAYESIAN,
            min_confidence_threshold=0.5,
            enable_bayesian_fusion=False
        )
        
        self.assertEqual(engine.strategy, FusionStrategy.BAYESIAN)
        self.assertEqual(engine.min_confidence_threshold, 0.5)
        self.assertFalse(engine.enable_bayesian_fusion)
    
    def test_extract_amendments_vision(self):
        """Test extracting amendments from vision result."""
        # Create mock vision result with amendments attribute
        vision_result = Mock()
        amendment1 = Mock()
        amendment2 = Mock()
        vision_result.amendments = [amendment1, amendment2]
        
        amendments = self.engine._extract_amendments(vision_result, "vision")
        
        self.assertEqual(amendments, [amendment1, amendment2])
    
    def test_extract_amendments_regex(self):
        """Test extracting amendments from regex result."""
        # Create mock regex result (amendments directly in result)
        regex_result = [Mock(), Mock()]
        
        amendments = self.engine._extract_amendments(regex_result, "regex")
        
        self.assertEqual(amendments, regex_result)
    
    def test_match_amendments_exact(self):
        """Test matching amendments with exact matches."""
        # Create mock amendments
        vision_amendments = []
        regex_amendments = []
        
        for i in range(3):
            vision_amd = Mock()
            vision_amd.section_number = f"45{i}"
            vision_amd.operation = "insert"
            vision_amendments.append(vision_amd)
            
            regex_amd = Mock()
            regex_amd.section_number = f"45{i}"
            regex_amd.operation = "insert"
            regex_amendments.append(regex_amd)
        
        # Mock the similarity calculation
        with patch.object(self.engine, '_calculate_similarity', return_value=0.9):
            pairs, unmatched_vision, unmatched_regex = self.engine._match_amendments(
                vision_amendments, regex_amendments
            )
        
        self.assertEqual(len(pairs), 3)
        self.assertEqual(len(unmatched_vision), 0)
        self.assertEqual(len(unmatched_regex), 0)
        
        for pair in pairs:
            self.assertGreaterEqual(pair.similarity_score, 0.0)
    
    def test_match_amendments_partial(self):
        """Test matching amendments with partial matches."""
        vision_amendments = [Mock(section_number="45A"), Mock(section_number="46B")]
        regex_amendments = [Mock(section_number="45A")]  # Only one matches
        
        with patch.object(self.engine, '_calculate_similarity', side_effect=[0.9, 0.3]):
            pairs, unmatched_vision, unmatched_regex = self.engine._match_amendments(
                vision_amendments, regex_amendments
            )
        
        self.assertEqual(len(pairs), 1)  # One match
        self.assertEqual(len(unmatched_vision), 1)  # One vision unmatched
        self.assertEqual(len(unmatched_regex), 0)  # All regex matched
    
    def test_fuse_agreeing_pair(self):
        """Test fusing agreeing amendment pair."""
        vision_amendment = Mock()
        vision_amendment.section_number = "45A"
        vision_amendment.operation = "insert"
        vision_amendment.text_content = "Insert new section"
        vision_amendment.confidence = 0.9
        
        regex_amendment = Mock()
        regex_amendment.section_number = "45A"
        regex_amendment.operation = "insert"
        regex_amendment.text_content = "Insert new section"
        regex_amendment.confidence = 0.8
        
        pair = AmendmentPair(
            vision_amendment=vision_amendment,
            regex_amendment=regex_amendment,
            similarity_score=0.95,
            conflicting_fields=[]
        )
        
        fused = self.engine._fuse_agreeing_pair(pair)
        
        self.assertIsNotNone(fused)
        # Should have combined confidence or other fused attributes
        # The actual implementation would have more specific checks
    
    def test_fuse_conflicting_pair(self):
        """Test fusing conflicting amendment pair."""
        vision_amendment = Mock()
        vision_amendment.section_number = "45A"
        vision_amendment.operation = "insert"
        vision_amendment.confidence = 0.9
        
        regex_amendment = Mock()
        regex_amendment.section_number = "45B"  # Different section
        regex_amendment.operation = "insert"
        regex_amendment.confidence = 0.8
        
        pair = AmendmentPair(
            vision_amendment=vision_amendment,
            regex_amendment=regex_amendment,
            similarity_score=0.6,
            conflicting_fields=["section_number"]
        )
        
        fused = self.engine._fuse_conflicting_pair(pair)
        
        self.assertIsNotNone(fused)
        # Should handle conflict resolution
        # The actual implementation would have more specific checks
    
    def test_handle_unmatched(self):
        """Test handling unmatched amendments."""
        unmatched = [Mock(confidence=0.7), Mock(confidence=0.4)]
        
        # With confidence threshold of 0.3, both should be included
        handled = self.engine._handle_unmatched(unmatched, "vision")
        
        self.assertEqual(len(handled), 2)
    
    def test_get_amendment_confidence(self):
        """Test getting amendment confidence."""
        amendment = Mock()
        amendment.confidence = 0.85
        
        confidence = self.engine._get_amendment_confidence(amendment)
        
        self.assertEqual(confidence, 0.85)
    
    def test_get_amendment_confidence_missing(self):
        """Test getting amendment confidence when missing."""
        amendment = Mock()
        # No confidence attribute
        
        confidence = self.engine._get_amendment_confidence(amendment)
        
        # Should return default
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_calculate_bayesian_confidence(self):
        """Test Bayesian confidence calculation."""
        vision_amendment = Mock()
        vision_amendment.confidence = 0.9
        
        regex_amendment = Mock()
        regex_amendment.confidence = 0.8
        
        # Mock the agreement check
        with patch.object(self.engine, '_amendments_agree', return_value=True):
            confidence = self.engine._calculate_bayesian_confidence(
                vision_amendment, regex_amendment
            )
        
        # Bayesian calculation should produce a value between 0 and 1
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


if __name__ == '__main__':
    unittest.main()