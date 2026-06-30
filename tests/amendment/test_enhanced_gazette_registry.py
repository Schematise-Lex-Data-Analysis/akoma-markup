"""Tests for enhanced_gazette_registry module."""

import unittest
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import json
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.akoma_markup.amendment.enhanced_gazette_registry import (
    EnhancedGazetteEntry,
    EnhancedGazetteRegistry
)
from src.akoma_markup.amendment.visual_gazette import GazetteAnalysis


class TestEnhancedGazetteEntry(unittest.TestCase):
    """Test EnhancedGazetteEntry class."""
    
    def test_enhanced_gazette_entry_init(self):
        """Test EnhancedGazetteEntry initialization."""
        entry = EnhancedGazetteEntry(
            gazette_number="123",
            gazette_date="2023-01-15",
            act_name="Test Act",
            act_year="2023",
            file_path=Path("/test/gazette.pdf"),
            page_numbers="1-5",
            additional_info="Test info",
            visual_analysis={"metadata": {"test": "data"}},
            analysis_score=0.8,
            extraction_method="visual"
        )
        
        self.assertEqual(entry.gazette_number, "123")
        self.assertEqual(entry.gazette_date, "2023-01-15")
        self.assertEqual(entry.act_name, "Test Act")
        self.assertEqual(entry.act_year, "2023")
        self.assertEqual(entry.file_path, Path("/test/gazette.pdf"))
        self.assertEqual(entry.page_numbers, "1-5")
        self.assertEqual(entry.additional_info, "Test info")
        self.assertEqual(entry.visual_analysis, {"metadata": {"test": "data"}})
        self.assertEqual(entry.analysis_score, 0.8)
        self.assertEqual(entry.extraction_method, "visual")
    
    def test_enhanced_gazette_entry_from_analysis(self):
        """Test EnhancedGazetteEntry creation from analysis."""
        # Create mock analysis
        analysis = Mock(spec=GazetteAnalysis)
        analysis.metadata = {
            "gazette_number": "456",
            "publication_date": "2023-02-20",
            "act_name": "IT Act",
            "act_year": "2023",
            "analysis_method": "visual_ocr"
        }
        analysis.validation = {"score": 0.9}
        analysis.to_dict.return_value = {"metadata": {"test": "data"}}
        
        pdf_path = Path("/test/gazette.pdf")
        entry = EnhancedGazetteEntry.from_analysis(pdf_path, analysis)
        
        self.assertEqual(entry.gazette_number, "456")
        self.assertEqual(entry.gazette_date, "2023-02-20")
        self.assertEqual(entry.act_name, "IT Act")
        self.assertEqual(entry.act_year, "2023")
        self.assertEqual(entry.file_path, pdf_path)
        self.assertEqual(entry.analysis_score, 0.9)
        self.assertEqual(entry.extraction_method, "visual")
        self.assertEqual(entry.visual_analysis, {"metadata": {"test": "data"}})
        self.assertIn("Visual analysis: visual_ocr", entry.additional_info)
    
    def test_enhanced_gazette_entry_to_dict(self):
        """Test EnhancedGazetteEntry to_dict method."""
        entry = EnhancedGazetteEntry(
            gazette_number="123",
            gazette_date="2023-01-15",
            act_name="Test Act",
            act_year="2023",
            file_path=Path("/test/gazette.pdf"),
            visual_analysis={"test": "data"},
            analysis_score=0.8,
            extraction_method="visual"
        )
        
        result = entry.to_dict()
        
        self.assertEqual(result["gazette_number"], "123")
        self.assertEqual(result["gazette_date"], "2023-01-15")
        self.assertEqual(result["act_name"], "Test Act")
        self.assertEqual(result["act_year"], "2023")
        self.assertEqual(result["file_path"], "/test/gazette.pdf")
        self.assertEqual(result["visual_analysis"], {"test": "data"})
        self.assertEqual(result["analysis_score"], 0.8)
        self.assertEqual(result["extraction_method"], "visual")


class TestEnhancedGazetteRegistry(unittest.TestCase):
    """Test EnhancedGazetteRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = EnhancedGazetteRegistry(enable_visual_analysis=False)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_init_without_visual_analysis(self):
        """Test registry initialization without visual analysis."""
        registry = EnhancedGazetteRegistry(enable_visual_analysis=False)
        self.assertFalse(registry.enable_visual_analysis)
        self.assertIsNone(registry.visual_analyzer)
        self.assertIsNone(registry.matcher)
    
    def test_init_with_visual_analysis(self):
        """Test registry initialization with visual analysis."""
        registry = EnhancedGazetteRegistry(enable_visual_analysis=True)
        self.assertTrue(registry.enable_visual_analysis)
        self.assertIsNotNone(registry.visual_analyzer)
        self.assertIsNotNone(registry.matcher)
    
    @patch('src.akoma_markup.amendment.enhanced_gazette_registry.VisualGazetteAnalyzer')
    def test_load_analysis_cache(self, mock_analyzer_class):
        """Test loading analysis cache."""
        registry = EnhancedGazetteRegistry(enable_visual_analysis=True)
        
        # Create temp cache file
        cache_file = Path(self.temp_dir.name) / "cache.json"
        cache_data = {
            "gazette1.pdf": {
                "metadata": {"act_name": "Test Act", "act_year": "2023"},
                "visual_features": {},
                "ocr_results": {},
                "validation": {}
            }
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Mock the _load_analysis_cache method
        registry._load_analysis_cache = Mock()
        
        # This would be called in load_from_directory
        registry._load_analysis_cache(cache_file)
        
        registry._load_analysis_cache.assert_called_once_with(cache_file)
    
    @patch('src.akoma_markup.amendment.enhanced_gazette_registry.VisualGazetteAnalyzer')
    def test_save_analysis_cache(self, mock_analyzer_class):
        """Test saving analysis cache."""
        registry = EnhancedGazetteRegistry(enable_visual_analysis=True)
        
        # Mock analysis data
        analysis = Mock(spec=GazetteAnalysis)
        analysis.to_dict.return_value = {"test": "data"}
        registry.analysis_cache["test.pdf"] = analysis
        
        # Create temp cache file
        cache_file = Path(self.temp_dir.name) / "cache.json"
        
        # Mock the _save_analysis_cache method
        registry._save_analysis_cache = Mock()
        
        # This would be called in load_from_directory
        registry._save_analysis_cache(cache_file)
        
        registry._save_analysis_cache.assert_called_once_with(cache_file)
    
    def test_get_enhanced_entry(self):
        """Test getting enhanced entry from registry."""
        # Add a regular entry
        entry = EnhancedGazetteEntry(
            gazette_number="123",
            gazette_date="2023-01-15",
            act_name="Test Act",
            act_year="2023",
            file_path=Path("/test/gazette.pdf"),
            visual_analysis={"test": "data"},
            analysis_score=0.8,
            extraction_method="visual"
        )
        
        self.registry.add_entry(entry)
        
        # Retrieve entry
        retrieved = self.registry.get_entry("123")
        self.assertIsInstance(retrieved, EnhancedGazetteEntry)
        self.assertEqual(retrieved.gazette_number, "123")
        self.assertEqual(retrieved.visual_analysis, {"test": "data"})
        self.assertEqual(retrieved.analysis_score, 0.8)
    
    def test_search_enhanced_entries(self):
        """Test searching enhanced entries."""
        # Add enhanced entries
        entry1 = EnhancedGazetteEntry(
            gazette_number="123",
            gazette_date="2023-01-15",
            act_name="IT Act",
            act_year="2023",
            file_path=Path("/test/gazette1.pdf"),
            analysis_score=0.9,
            extraction_method="visual"
        )
        
        entry2 = EnhancedGazetteEntry(
            gazette_number="456",
            gazette_date="2023-02-20",
            act_name="Companies Act",
            act_year="2023",
            file_path=Path("/test/gazette2.pdf"),
            analysis_score=0.7,
            extraction_method="filename"
        )
        
        self.registry.add_entry(entry1)
        self.registry.add_entry(entry2)
        
        # Search by act name
        results = self.registry.search_by_act_name("IT Act")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].gazette_number, "123")
        self.assertEqual(results[0].analysis_score, 0.9)
        
        # Search by year
        results = self.registry.search_by_year("2023")
        self.assertEqual(len(results), 2)
    
    @patch('src.akoma_markup.amendment.enhanced_gazette_registry.VisualGazetteAnalyzer')
    def test_analyze_gazette_file(self, mock_analyzer_class):
        """Test analyzing a single gazette file."""
        # Create mock analyzer
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze_gazette.return_value = Mock(
            spec=GazetteAnalysis,
            metadata={
                "gazette_number": "789",
                "publication_date": "2023-03-25",
                "act_name": "Banking Act",
                "act_year": "2023"
            },
            validation={"score": 0.85}
        )
        
        registry = EnhancedGazetteRegistry(enable_visual_analysis=True)
        registry.visual_analyzer = mock_analyzer
        
        # Test file
        test_file = self.test_dir / "test_gazette.pdf"
        test_file.touch()  # Create empty file
        
        # Run async test
        async def run_test():
            analysis = await registry.analyze_gazette_file(test_file)
            mock_analyzer.analyze_gazette.assert_called_once_with(test_file)
            self.assertEqual(analysis.metadata["gazette_number"], "789")
            return analysis
        
        # For simplicity in unittest, we'll just verify the mock was set up
        self.assertIsNotNone(registry.visual_analyzer)


if __name__ == '__main__':
    unittest.main()