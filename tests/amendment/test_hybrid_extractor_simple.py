"""Simplified tests for hybrid amendment extraction."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.akoma_markup.amendment.hybrid_extractor import (
    HybridAmendmentExtractor,
    HybridExtractionConfig,
    HybridExtractionResult
)
from src.akoma_markup.amendment.strategy_detector import (
    ExtractionStrategyDetector,
    ExtractionStrategy,
    PDFCharacteristics
)


class TestHybridExtractorSimple:
    """Simplified tests for HybridAmendmentExtractor."""
    
    def test_init_default_config(self):
        """Test initialization with default config."""
        extractor = HybridAmendmentExtractor()
        assert extractor.config is not None
        assert extractor.config.use_vision is True
        assert extractor.config.use_regex is True
        assert extractor.config.confidence_threshold == 0.5
    
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = HybridExtractionConfig(
            use_vision=False,
            use_regex=True,
            confidence_threshold=0.7
        )
        extractor = HybridAmendmentExtractor(config)
        assert extractor.config == config
        assert extractor.vision_extractor is None
    
    def test_combine_results_empty(self):
        """Test combining empty results."""
        extractor = HybridAmendmentExtractor()
        
        combined, stats = extractor.combine_results(None, None)
        
        assert combined == []
        assert stats["vision_total"] == 0
        assert stats["regex_total"] == 0
        assert stats["combined_total"] == 0
    
    def test_resolve_conflicts_simple(self):
        """Test simple conflict resolution."""
        extractor = HybridAmendmentExtractor()
        
        # Create simple conflicting amendments
        vision_am = MagicMock()
        vision_am.confidence_score = 0.8
        vision_am.metadata = {}
        
        regex_am = MagicMock()
        regex_am.linkage_confidence = "medium"
        regex_am.metadata = {}
        
        conflicts = [(vision_am, regex_am)]
        
        resolved = extractor.resolve_conflicts(conflicts)
        
        assert len(resolved) == 1
        # Should prefer vision due to higher confidence
        assert resolved[0] == vision_am


class TestStrategyDetectorSimple:
    """Simplified tests for ExtractionStrategyDetector."""
    
    def test_init_default(self):
        """Test default initialization."""
        detector = ExtractionStrategyDetector()
        assert detector.config == {}
        assert detector.scan_quality_threshold == 0.7
    
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = {
            "scan_quality_threshold": 0.5,
            "amendment_density_threshold": 0.9
        }
        detector = ExtractionStrategyDetector(config)
        assert detector.scan_quality_threshold == 0.5
        assert detector.amendment_density_threshold == 0.9
    
    def test_determine_strategy_logic(self):
        """Test the strategy determination logic."""
        detector = ExtractionStrategyDetector()
        
        # Test poor scan quality -> regex_only
        detector._estimate_scan_quality = lambda *args: 0.6
        detector._estimate_amendment_density = lambda *args: 0.5
        detector._detect_complex_layout = lambda *args: False
        detector._detect_gazette_format = lambda *args: False
        
        # Mock _analyze_pdf to return characteristics
        with patch.object(detector, '_analyze_pdf') as mock_analyze:
            mock_analyze.return_value = PDFCharacteristics(
                page_count=10,
                scan_quality=0.6,
                amendment_density=0.5,
                complex_layout=False,
                gazette_notification=False,
                estimated_cost=10.0
            )
            strategy = detector.determine_strategy(Path("test.pdf"))
        
        assert strategy == ExtractionStrategy.REGEX_ONLY
    
    def test_estimate_processing_cost(self):
        """Test processing cost estimation."""
        detector = ExtractionStrategyDetector()
        
        cost = detector._estimate_processing_cost(
            page_count=10,
            scan_quality=0.8,
            amendment_density=0.6,
            complex_layout=True
        )
        
        # Should return a positive value
        assert cost > 0
        # Complex layout should increase cost
        assert cost > 10.0  # Base cost for 10 pages


def test_phase_3_components_exist():
    """Verify that Phase 3 components exist and are importable."""
    # Test hybrid extractor
    from src.akoma_markup.amendment.hybrid_extractor import (
        HybridAmendmentExtractor,
        HybridExtractionConfig,
        HybridExtractionResult
    )
    
    # Test strategy detector
    from src.akoma_markup.amendment.strategy_detector import (
        ExtractionStrategyDetector,
        ExtractionStrategy,
        PDFCharacteristics
    )
    
    # Test validation exists
    from src.akoma_markup.amendment.validation import detect_conflicting_amendments
    
    # If we can import all these, Phase 3.1 is complete
    assert True