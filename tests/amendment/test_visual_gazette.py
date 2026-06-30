"""Tests for visual gazette analysis module."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.akoma_markup.amendment.visual_gazette import (
    VisualGazetteAnalyzer,
    GazetteAnalysis,
    VisualFeatures,
    create_visual_gazette_analyzer,
    analyze_gazette_directory
)


class TestVisualFeatures(unittest.TestCase):
    """Test VisualFeatures class."""
    
    def test_visual_features_init(self):
        """Test VisualFeatures initialization."""
        features = VisualFeatures()
        
        self.assertFalse(features.has_official_seal)
        self.assertFalse(features.has_letterhead)
        self.assertFalse(features.has_signatures)
        self.assertEqual(features.visual_complexity_score, 0.0)
        self.assertEqual(features.font_analysis, {})
        self.assertEqual(features.page_numbering, {})
        self.assertEqual(features.color_usage, {})
        self.assertEqual(features.margin_analysis, {})
    
    def test_visual_features_to_dict(self):
        """Test VisualFeatures to_dict method."""
        features = VisualFeatures()
        features.has_official_seal = True
        features.visual_complexity_score = 0.8
        
        result = features.to_dict()
        
        self.assertTrue(result["has_official_seal"])
        self.assertEqual(result["visual_complexity_score"], 0.8)
        self.assertEqual(result["font_analysis"], {})
        self.assertEqual(result["page_numbering"], {})


class TestGazetteAnalysis(unittest.TestCase):
    """Test GazetteAnalysis class."""
    
    def test_gazette_analysis_init(self):
        """Test GazetteAnalysis initialization."""
        pdf_path = Path("/test/gazette.pdf")
        analysis = GazetteAnalysis(pdf_path)
        
        self.assertEqual(analysis.pdf_path, pdf_path)
        self.assertEqual(analysis.metadata, {})
        self.assertEqual(analysis.visual_features, {})
        self.assertEqual(analysis.ocr_results, {})
        self.assertEqual(analysis.structure, {})
        self.assertEqual(analysis.validation, {})
    
    def test_gazette_analysis_to_dict(self):
        """Test GazetteAnalysis to_dict method."""
        pdf_path = Path("/test/gazette.pdf")
        analysis = GazetteAnalysis(
            pdf_path,
            metadata={"gazette_number": "123456", "act_year": "2023"}
        )
        analysis.visual_features = {"has_official_seal": True}
        analysis.ocr_results = {1: "Page 1 text"}
        
        result = analysis.to_dict()
        
        self.assertEqual(result["pdf_path"], str(pdf_path))
        self.assertEqual(result["metadata"]["gazette_number"], "123456")
        self.assertEqual(result["metadata"]["act_year"], "2023")
        self.assertTrue(result["visual_features"]["has_official_seal"])
        self.assertEqual(result["ocr_results"]["1"], "Page 1 text")
    
    def test_gazette_analysis_from_dict(self):
        """Test GazetteAnalysis from_dict method."""
        data = {
            "pdf_path": "/test/gazette.pdf",
            "metadata": {"gazette_number": "123456"},
            "visual_features": {"has_official_seal": True},
            "ocr_results": {"1": "Page 1 text"},
            "structure": {"estimated_sections": 3},
            "validation": {"valid": True}
        }
        
        analysis = GazetteAnalysis.from_dict(data)
        
        self.assertEqual(str(analysis.pdf_path), "/test/gazette.pdf")
        self.assertEqual(analysis.metadata["gazette_number"], "123456")
        self.assertTrue(analysis.visual_features["has_official_seal"])
        self.assertEqual(analysis.ocr_results[1], "Page 1 text")
        self.assertEqual(analysis.structure["estimated_sections"], 3)
        self.assertTrue(analysis.validation["valid"])


class TestVisualGazetteAnalyzer(unittest.TestCase):
    """Test VisualGazetteAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = VisualGazetteAnalyzer(enable_ocr=False, cache_results=False)
        self.test_pdf = Path("/test/gazette_2023.pdf")
    
    def test_init(self):
        """Test analyzer initialization."""
        self.assertFalse(self.analyzer.enable_ocr)
        self.assertFalse(self.analyzer.cache_results)
        self.assertEqual(self.analyzer._analysis_cache, {})
    
    @patch.object(VisualGazetteAnalyzer, '_extract_visual_features')
    @patch.object(VisualGazetteAnalyzer, '_extract_metadata')
    @patch.object(VisualGazetteAnalyzer, '_analyze_gazette_structure')
    @patch.object(VisualGazetteAnalyzer, '_validate_gazette')
    async def test_analyze_gazette_success(self, mock_validate, mock_structure, 
                                           mock_metadata, mock_features):
        """Test successful gazette analysis."""
        # Setup mocks
        mock_features.return_value = VisualFeatures()
        mock_metadata.return_value = {"gazette_number": "123456", "act_year": "2023"}
        mock_structure.return_value = {"estimated_sections": 3}
        mock_validate.return_value = {"valid": True, "score": 0.9}
        
        # Run analysis
        analysis = await self.analyzer.analyze_gazette(self.test_pdf)
        
        # Verify results
        self.assertEqual(analysis.pdf_path, self.test_pdf)
        self.assertEqual(analysis.metadata["gazette_number"], "123456")
        self.assertEqual(analysis.metadata["act_year"], "2023")
        self.assertEqual(analysis.structure, {"estimated_sections": 3})
        self.assertEqual(analysis.validation, {"valid": True, "score": 0.9})
        
        # Verify method calls
        mock_features.assert_called_once_with(self.test_pdf)
        mock_metadata.assert_called_once()
        mock_structure.assert_called_once_with(self.test_pdf)
        mock_validate.assert_called_once()
    
    async def test_analyze_gazette_cache(self):
        """Test gazette analysis with caching."""
        analyzer = VisualGazetteAnalyzer(cache_results=True)
        
        # First call should analyze
        with patch.object(analyzer, '_extract_visual_features') as mock_features:
            mock_features.return_value = VisualFeatures()
            analysis1 = await analyzer.analyze_gazette(self.test_pdf)
        
        # Second call should use cache
        analysis2 = await analyzer.analyze_gazette(self.test_pdf)
        
        self.assertEqual(analysis1, analysis2)
        self.assertEqual(len(analyzer._analysis_cache), 1)
    
    async def test_analyze_gazette_error(self):
        """Test gazette analysis with error."""
        # Mock an exception
        with patch.object(self.analyzer, '_extract_visual_features', 
                         side_effect=Exception("Analysis failed")):
            analysis = await self.analyzer.analyze_gazette(self.test_pdf)
        
        # Should return analysis with error metadata
        self.assertEqual(analysis.pdf_path, self.test_pdf)
        self.assertIn("error", analysis.metadata)
        self.assertEqual(analysis.metadata["filename"], self.test_pdf.name)
    
    def test_extract_date_from_filename(self):
        """Test date extraction from filename."""
        test_cases = [
            ("Gazette_2023-12-15.pdf", "2023-12-15"),
            ("Gazette_15-12-2023.pdf", "2023-12-15"),
            ("2023_Gazette.pdf", "2023-01-01"),
            ("Gazette_no_date.pdf", None),
        ]
        
        for filename, expected_date in test_cases:
            with self.subTest(filename=filename):
                date = self.analyzer._extract_date_from_filename(filename)
                self.assertEqual(date, expected_date)
    
    def test_extract_act_info_from_filename(self):
        """Test act information extraction from filename."""
        test_cases = [
            ("act_10_of_2009.pdf", {"act_number": "10", "act_year": "2009", 
                                   "act_reference": "Act 10 of 2009"}),
            ("it_act_rules_2023.pdf", {"act_name": "Information Technology Act"}),
            ("intermediary_guidelines_2021.pdf", 
             {"act_name": "Intermediary Guidelines and Digital Media Ethics Code Rules"}),
            ("Generic_Amendment.pdf", {}),
        ]
        
        for filename, expected_info in test_cases:
            with self.subTest(filename=filename):
                info = self.analyzer._extract_act_info_from_filename(filename)
                
                for key, expected_value in expected_info.items():
                    self.assertEqual(info.get(key), expected_value)
    
    def test_validate_gazette(self):
        """Test gazette validation."""
        test_cases = [
            # Valid metadata
            (
                {"act_year": "2023", "act_name": "IT Act", 
                 "publication_date": "2023-12-15", "gazette_number": "123456"},
                {"valid": True, "issues": [], "score": 1.0}
            ),
            # Missing fields
            (
                {"act_year": "2023"},
                {"valid": False, "issues": ["missing_act_name", "missing_publication_date"]}
            ),
            # Invalid year
            (
                {"act_year": "1799", "act_name": "IT Act", "publication_date": "2023-12-15"},
                {"valid": False, "issues": ["invalid_act_year"]}
            ),
        ]
        
        for metadata, expected in test_cases:
            with self.subTest(metadata=metadata):
                result = self.analyzer._validate_gazette(metadata, {})
                
                self.assertEqual(result["valid"], expected["valid"])
                
                if "issues" in expected:
                    self.assertEqual(set(result["issues"]), set(expected["issues"]))
                
                if "score" in expected:
                    self.assertAlmostEqual(result["score"], expected["score"], places=1)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def test_create_visual_gazette_analyzer(self):
        """Test create_visual_gazette_analyzer function."""
        analyzer = create_visual_gazette_analyzer(enable_ocr=True, cache_results=False)
        
        self.assertIsInstance(analyzer, VisualGazetteAnalyzer)
        self.assertTrue(analyzer.enable_ocr)
        self.assertFalse(analyzer.cache_results)
    
    @patch('src.akoma_markup.amendment.visual_gazette.VisualGazetteAnalyzer')
    async def test_analyze_gazette_directory(self, mock_analyzer_class):
        """Test analyze_gazette_directory function."""
        # Setup mock
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze_gazette.return_value = GazetteAnalysis(
            Path("test.pdf"),
            metadata={"test": "data"}
        )
        mock_analyzer_class.return_value = mock_analyzer
        
        # Create temp directory with dummy files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy PDF files
            (temp_path / "gazette1.pdf").write_text("dummy")
            (temp_path / "gazette2.pdf").write_text("dummy")
            
            # Run analysis
            with patch('src.akoma_markup.amendment.visual_gazette.logger'):
                analyses = await analyze_gazette_directory(temp_path)
            
            # Verify results
            self.assertEqual(len(analyses), 2)
            self.assertIn(str(temp_path / "gazette1.pdf"), analyses)
            self.assertIn(str(temp_path / "gazette2.pdf"), analyses)
            
            # Verify analyzer was called for each file
            self.assertEqual(mock_analyzer.analyze_gazette.call_count, 2)


if __name__ == '__main__':
    unittest.main()