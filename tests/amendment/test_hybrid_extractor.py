"""Tests for hybrid amendment extraction."""

import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

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


class TestHybridExtractor:
    """Tests for HybridAmendmentExtractor."""
    
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
    
    @pytest.mark.asyncio
    async def test_extract_vision_only(self):
        """Test extraction with vision only."""
        config = HybridExtractionConfig(
            use_vision=True,
            use_regex=False,
            vision_fallback_to_regex=False
        )
        
        extractor = HybridAmendmentExtractor(config)
        
        # Mock the vision extractor
        mock_vision_result = Mock()
        # Create proper mock amendment objects with required attributes
        vision_amendment1 = Mock()
        vision_amendment1.page_num = 1
        vision_amendment1.act_number = "13"
        vision_amendment1.act_year = "2015"
        vision_amendment1.section_number = "5"
        vision_amendment1.amendment_type = "insert"
        vision_amendment1.target_location = "after"
        vision_amendment1.original_text = "Insert new subsection (1A)"
        
        vision_amendment2 = Mock()
        vision_amendment2.page_num = 2
        vision_amendment2.act_number = "13"
        vision_amendment2.act_year = "2015"
        vision_amendment2.section_number = "8"
        vision_amendment2.amendment_type = "replace"
        vision_amendment2.target_location = ""
        vision_amendment2.original_text = "Replace 'shall' with 'may'"
        
        mock_vision_result.extracted_amendments = [vision_amendment1, vision_amendment2]
        
        with patch.object(extractor.vision_extractor, 'extract_from_pdf', 
                         AsyncMock(return_value=mock_vision_result)):
            result = await extractor.extract(Path("test.pdf"))
            
        assert isinstance(result, HybridExtractionResult)
        assert result.total_amendments == 2
        assert result.vision_result == mock_vision_result
        assert result.regex_result is None
    
    @pytest.mark.asyncio
    async def test_extract_regex_only(self):
        """Test extraction with regex only."""
        config = HybridExtractionConfig(
            use_vision=False,
            use_regex=True
        )
        
        extractor = HybridAmendmentExtractor(config)
        
        # Mock the regex extractor (sync function)
        mock_regex_result = Mock()
        # Create proper mock amendment objects with required attributes
        regex_amendment1 = Mock()
        regex_amendment1.page_num = 1
        regex_amendment1.act_number = "13"
        regex_amendment1.target_section = "5"
        regex_amendment1.amendment_type = "insert"
        regex_amendment1.text = "Insert new subsection (1A)"
        
        regex_amendment2 = Mock()
        regex_amendment2.page_num = 2
        regex_amendment2.act_number = "13"
        regex_amendment2.target_section = "8"
        regex_amendment2.amendment_type = "replace"
        regex_amendment2.text = "Replace 'shall' with 'may'"
        
        regex_amendment3 = Mock()
        regex_amendment3.page_num = 3
        regex_amendment3.act_number = "13"
        regex_amendment3.target_section = "12"
        regex_amendment3.amendment_type = "delete"
        regex_amendment3.text = "Delete subsection (3)"
        
        mock_regex_result.amendments = [regex_amendment1, regex_amendment2, regex_amendment3]
        
        with patch('src.akoma_markup.amendment.hybrid_extractor.extract_amendments_from_pdf',
                  return_value=mock_regex_result):
            result = await extractor.extract(Path("test.pdf"))
            
        assert isinstance(result, HybridExtractionResult)
        assert result.total_amendments == 3
        assert result.vision_result is None
        assert result.regex_result == mock_regex_result
    
    @pytest.mark.asyncio
    async def test_extract_hybrid_both_success(self):
        """Test hybrid extraction with both methods succeeding."""
        config = HybridExtractionConfig(
            use_vision=True,
            use_regex=True
        )
        
        extractor = HybridAmendmentExtractor(config)
        
        # Mock vision extractor
        mock_vision_result = Mock()
        vision_amendment = Mock()
        vision_amendment.page_num = 1
        vision_amendment.section = "5"
        vision_amendment.act_number = "13"
        vision_amendment.act_year = "2015"
        mock_vision_result.extracted_amendments = [vision_amendment]
        
        # Mock regex extractor
        mock_regex_result = Mock()
        regex_amendment = Mock()
        regex_amendment.page_num = 1
        regex_amendment.section = "5"
        regex_amendment.act_number = "13"
        regex_amendment.act_year = "2015"
        mock_regex_result.amendments = [regex_amendment]
        
        with patch.object(extractor.vision_extractor, 'extract_from_pdf',
                         AsyncMock(return_value=mock_vision_result)), \
             patch('src.akoma_markup.amendment.hybrid_extractor.extract_amendments_from_pdf',
                  return_value=mock_regex_result):
            
            result = await extractor.extract(Path("test.pdf"))
            
        assert isinstance(result, HybridExtractionResult)
        assert result.total_amendments == 1  # Should be deduplicated
        assert result.combination_stats["both_methods"] == 1
    
    def test_combine_results_empty(self):
        """Test combining empty results."""
        extractor = HybridAmendmentExtractor()
        
        combined, stats = extractor.combine_results(None, None)
        
        assert combined == []
        assert stats["vision_total"] == 0
        assert stats["regex_total"] == 0
        assert stats["combined_total"] == 0
    
    def test_resolve_conflicts(self):
        """Test conflict resolution between amendments."""
        extractor = HybridAmendmentExtractor()
        
        # Create conflicting amendments
        vision_am = Mock()
        vision_am.confidence_score = 0.8
        vision_am.metadata = None
        
        regex_am = Mock()
        regex_am.linkage_confidence = "medium"
        regex_am.metadata = None
        
        conflicts = [(vision_am, regex_am)]
        
        resolved = extractor.resolve_conflicts(conflicts)
        
        assert len(resolved) == 1
        # Should prefer vision due to higher confidence
        assert resolved[0] == vision_am


class TestStrategyDetector:
    """Tests for ExtractionStrategyDetector."""
    
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
    
    @patch('src.akoma_markup.amendment.strategy_detector.pdfplumber')
    def test_determine_strategy_poor_scan(self, mock_pdfplumber):
        """Test strategy determination for poor scan quality."""
        detector = ExtractionStrategyDetector({"scan_quality_threshold": 0.7})
        
        # Mock PDF with poor scan quality
        mock_pdf = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = "short text"  # Low text density
        mock_page.images = ["image1", "image2"]  # Has images
        mock_pdf.pages = [mock_page]
        
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        with patch.object(detector, '_estimate_scan_quality', return_value=0.6):
            strategy = detector.determine_strategy(Path("test.pdf"))
        
        assert strategy == ExtractionStrategy.REGEX_ONLY
    
    @patch('src.akoma_markup.amendment.strategy_detector.pdfplumber')
    def test_determine_strategy_dense_amendments(self, mock_pdfplumber):
        """Test strategy determination for dense amendments."""
        detector = ExtractionStrategyDetector({"amendment_density_threshold": 0.8})
        
        # Mock PDF with high amendment density
        mock_pdf = Mock()
        mock_pdf.pages = [Mock()]
        
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        with patch.object(detector, '_estimate_scan_quality', return_value=0.8), \
             patch.object(detector, '_estimate_amendment_density', return_value=0.9):
            strategy = detector.determine_strategy(Path("test.pdf"))
        
        assert strategy == ExtractionStrategy.VISION_ONLY
    
    @patch('src.akoma_markup.amendment.strategy_detector.pdfplumber')
    def test_determine_strategy_complex_layout(self, mock_pdfplumber):
        """Test strategy determination for complex layout."""
        detector = ExtractionStrategyDetector()
        
        # Mock PDF with complex layout
        mock_pdf = Mock()
        mock_pdf.pages = [Mock()]
        
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        with patch.object(detector, '_estimate_scan_quality', return_value=0.8), \
             patch.object(detector, '_estimate_amendment_density', return_value=0.5), \
             patch.object(detector, '_detect_complex_layout', return_value=True):
            strategy = detector.determine_strategy(Path("test.pdf"))
        
        assert strategy == ExtractionStrategy.HYBRID_PARALLEL
    
    @patch('src.akoma_markup.amendment.strategy_detector.pdfplumber')
    def test_determine_strategy_gazette(self, mock_pdfplumber):
        """Test strategy determination for gazette notification."""
        detector = ExtractionStrategyDetector()
        
        # Mock PDF that appears to be a gazette
        mock_pdf = Mock()
        mock_pdf.pages = [Mock()]
        
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        with patch.object(detector, '_estimate_scan_quality', return_value=0.8), \
             patch.object(detector, '_estimate_amendment_density', return_value=0.5), \
             patch.object(detector, '_detect_complex_layout', return_value=False), \
             patch.object(detector, '_detect_gazette_format', return_value=True):
            strategy = detector.determine_strategy(Path("test.pdf"))
        
        assert strategy == ExtractionStrategy.VISION_ONLY
    
    def test_analyze_pdf_error_handling(self):
        """Test PDF analysis error handling."""
        detector = ExtractionStrategyDetector()
        
        with patch('src.akoma_markup.amendment.strategy_detector.pdfplumber.open', 
                  side_effect=Exception("PDF error")):
            characteristics = detector._analyze_pdf(Path("test.pdf"))
        
        # Should return default characteristics on error
        assert characteristics.page_count == 1
        assert characteristics.scan_quality == 0.8
    
    def test_estimate_scan_quality(self):
        """Test scan quality estimation."""
        detector = ExtractionStrategyDetector()
        
        # Mock PDF with reasonable quality
        mock_pdf = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = "This is a test document with reasonable amount of text for analysis."
        mock_page.images = []
        mock_pdf.pages = [mock_page, mock_page, mock_page]
        
        with patch('src.akoma_markup.amendment.strategy_detector.pdfplumber.open', 
                  return_value=mock_pdf):
            quality = detector._estimate_scan_quality(Path("test.pdf"), sample_pages=3)
        
        # Should return a reasonable value
        assert 0.0 <= quality <= 1.0
    
    def test_estimate_amendment_density(self):
        """Test amendment density estimation."""
        detector = ExtractionStrategyDetector()
        
        # Mock PDF with amendment keywords
        mock_pdf = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = "Amendment to section 5 of the Act insert new subsection delete old text"
        mock_pdf.pages = [mock_page, mock_page, mock_page]
        
        with patch('src.akoma_markup.amendment.strategy_detector.pdfplumber.open',
                  return_value=mock_pdf):
            density = detector._estimate_amendment_density(Path("test.pdf"), sample_pages=3)
        
        # Should detect amendment keywords
        assert 0.0 <= density <= 1.0
    
    def test_detect_complex_layout(self):
        """Test complex layout detection."""
        detector = ExtractionStrategyDetector()
        
        # Mock PDF with tables
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.find_tables.return_value = [MagicMock()]  # Has tables
        mock_page.chars = []
        mock_pdf.pages = [mock_page, mock_page, mock_page]
        
        # Create a proper context manager
        context_manager = MagicMock()
        context_manager.__enter__.return_value = mock_pdf
        context_manager.__exit__.return_value = None
        
        with patch('src.akoma_markup.amendment.strategy_detector.pdfplumber.open',
                  return_value=context_manager):
            is_complex = detector._detect_complex_layout(Path("test.pdf"), sample_pages=3)
        
        # Should detect complex layout due to tables
        assert is_complex is True
    
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