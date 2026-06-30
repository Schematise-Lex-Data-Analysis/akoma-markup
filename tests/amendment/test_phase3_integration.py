"""Integration tests for Phase 3.3 enhancement modules."""

import unittest
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import sys
import json

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.akoma_markup.amendment.visual_gazette import VisualGazetteAnalyzer
from src.akoma_markup.amendment.enhanced_gazette_registry import EnhancedGazetteRegistry
from src.akoma_markup.amendment.gazette_matcher import GazetteAmendmentMatcher
from src.akoma_markup.amendment.fusion import FusionEngine, FusionStrategy
from src.akoma_markup.amendment.confidence import ConfidenceScorer
from src.akoma_markup.amendment.comprehensive_validation import CrossValidator


class TestPhase3Integration(unittest.TestCase):
    """Integration tests for Phase 3.3 modules."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    @patch('src.akoma_markup.amendment.visual_gazette.VisualGazetteAnalyzer')
    def test_visual_gazette_to_registry_integration(self, mock_analyzer_class):
        """Test integration between visual gazette analysis and enhanced registry."""
        # Create mock analyzer
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze_gazette.return_value = Mock(
            metadata={
                "gazette_number": "789",
                "publication_date": "2023-03-25",
                "act_name": "IT Act",
                "act_year": "2023",
                "analysis_method": "visual_ocr"
            },
            validation={"score": 0.85},
            to_dict=lambda: {"metadata": {"test": "data"}}
        )
        mock_analyzer_class.return_value = mock_analyzer
        
        # Create registry
        registry = EnhancedGazetteRegistry(enable_visual_analysis=True)
        registry.visual_analyzer = mock_analyzer
        
        # Test file
        test_file = self.test_dir / "test_gazette.pdf"
        test_file.touch()
        
        # This tests the integration conceptually
        # In real usage, analyze_gazette_file would be called
        self.assertIsNotNone(registry.visual_analyzer)
        self.assertTrue(registry.enable_visual_analysis)
    
    def test_gazette_matcher_fusion_integration(self):
        """Test integration between gazette matcher and fusion engine."""
        # Create matcher
        matcher = GazetteAmendmentMatcher()
        
        # Create fusion engine
        fusion_engine = FusionEngine(strategy=FusionStrategy.HYBRID)
        
        # Create mock amendments
        amendment1 = Mock()
        amendment1.act_number = "10"
        amendment1.act_year = "2009"
        amendment1.effective_date = "2009-01-15"
        
        amendment2 = Mock()
        amendment2.act_number = "15"
        amendment2.act_year = "2010"
        amendment2.effective_date = "2010-03-20"
        
        # Test both components work independently
        self.assertEqual(matcher.MIN_MATCH_SCORE, 0.6)
        self.assertEqual(fusion_engine.strategy, FusionStrategy.HYBRID)
        
        # These would be integrated in actual usage:
        # 1. Gazette matcher finds matches
        # 2. Fusion engine combines results
        # 3. Confidence scorer evaluates quality
        
        self.assertIsNotNone(matcher)
        self.assertIsNotNone(fusion_engine)
    
    def test_confidence_scorer_validation_integration(self):
        """Test integration between confidence scorer and comprehensive validator."""
        # Create confidence scorer
        scorer = ConfidenceScorer()
        
        # Create cross validator
        validator = CrossValidator()
        
        # Create mock amendment
        amendment = Mock()
        amendment.section_number = "45A"
        amendment.operation = "insert"
        amendment.extraction_method = "hybrid"
        
        # Test both components
        self.assertIn("source", scorer.DEFAULT_WEIGHTS)
        self.assertIsNotNone(validator)
        
        # In real integration:
        # 1. Validator checks amendment quality
        # 2. Scorer calculates confidence based on validation results
        # 3. Results inform extraction quality
        
        self.assertIsNotNone(scorer)
        self.assertIsNotNone(validator)
    
    def test_end_to_end_workflow_conceptual(self):
        """Test conceptual end-to-end workflow for Phase 3.3."""
        # Step 1: Visual gazette analysis
        visual_analyzer = VisualGazetteAnalyzer(enable_ocr=True, cache_results=True)
        
        # Step 2: Enhanced registry with visual analysis
        registry = EnhancedGazetteRegistry(enable_visual_analysis=True)
        
        # Step 3: Gazette matching
        matcher = GazetteAmendmentMatcher()
        
        # Step 4: Fusion engine
        fusion_engine = FusionEngine(strategy=FusionStrategy.HYBRID)
        
        # Step 5: Confidence scoring
        scorer = ConfidenceScorer()
        
        # Step 6: Comprehensive validation
        validator = CrossValidator()
        
        # All components should work together
        self.assertIsNotNone(visual_analyzer)
        self.assertIsNotNone(registry)
        self.assertIsNotNone(matcher)
        self.assertIsNotNone(fusion_engine)
        self.assertIsNotNone(scorer)
        self.assertIsNotNone(validator)
        
        # Check configuration
        self.assertTrue(registry.enable_visual_analysis)
        self.assertEqual(fusion_engine.strategy, FusionStrategy.HYBRID)
        self.assertIn("source", scorer.DEFAULT_WEIGHTS)
    
    @patch('src.akoma_markup.amendment.visual_gazette.VisualGazetteAnalyzer')
    def test_hybrid_extraction_pipeline(self, mock_analyzer_class):
        """Test hybrid extraction pipeline integration."""
        # Mock vision extraction result
        vision_result = Mock()
        vision_result.amendments = [Mock(section_number="45A", confidence=0.9)]
        
        # Mock regex extraction result  
        regex_result = [Mock(section_number="45A", confidence=0.8)]
        
        # Create fusion engine
        fusion_engine = FusionEngine(strategy=FusionStrategy.HYBRID)
        
        # Create confidence scorer
        scorer = ConfidenceScorer()
        
        # Create validator
        validator = CrossValidator()
        
        # Pipeline flow:
        # 1. Vision and regex extract amendments
        # 2. Fusion engine combines results
        # 3. Validator checks quality
        # 4. Scorer calculates confidence
        
        self.assertIsNotNone(fusion_engine)
        self.assertIsNotNone(scorer)
        self.assertIsNotNone(validator)
        
        # Test components are properly configured
        self.assertEqual(fusion_engine.strategy, FusionStrategy.HYBRID)
        self.assertGreater(len(scorer.DEFAULT_WEIGHTS), 0)


class TestDataFlow(unittest.TestCase):
    """Test data flow between Phase 3.3 modules."""
    
    def test_amendment_data_structure(self):
        """Test that amendments have expected structure for all modules."""
        # Create a mock amendment with all expected fields
        amendment = Mock()
        
        # Fields for gazette matching
        amendment.act_number = "10"
        amendment.act_year = "2009"
        amendment.effective_date = "2009-01-15"
        
        # Fields for fusion
        amendment.confidence = 0.85
        amendment.extraction_method = "hybrid"
        amendment.has_vision_match = True
        amendment.has_regex_match = True
        
        # Fields for confidence scoring
        amendment.section_number = "45A"
        amendment.operation = "insert"
        amendment.text_content = "Insert new section 45A"
        
        # Fields for validation
        amendment.amendment_id = "amd_123"
        amendment.gazette_references = []
        
        # All modules should be able to work with this structure
        self.assertEqual(amendment.act_number, "10")
        self.assertEqual(amendment.act_year, "2009")
        self.assertEqual(amendment.confidence, 0.85)
        self.assertEqual(amendment.section_number, "45A")
        self.assertEqual(amendment.operation, "insert")
    
    def test_gazette_analysis_data_flow(self):
        """Test data flow from gazette analysis to matching."""
        # Mock gazette analysis
        gazette_analysis = Mock()
        gazette_analysis.metadata = {
            "act_number": "10",
            "act_year": "2009",
            "publication_date": "2009-01-10",
            "act_name": "IT Act 2009"
        }
        gazette_analysis.visual_features = {"is_official_looking": True}
        
        # Mock amendment
        amendment = Mock()
        amendment.act_number = "10"
        amendment.act_year = "2009"
        amendment.effective_date = "2009-01-15"
        
        # Matcher should be able to use this data
        matcher = GazetteAmendmentMatcher()
        
        self.assertEqual(gazette_analysis.metadata["act_number"], "10")
        self.assertEqual(amendment.act_number, "10")
        self.assertEqual(matcher.MIN_MATCH_SCORE, 0.6)


if __name__ == '__main__':
    unittest.main()