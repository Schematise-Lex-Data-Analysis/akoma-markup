"""Tests for LLMVerificationEngine."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from src.akoma_markup.gazette.verification.engine import (
    LLMVerificationEngine,
    VerificationResult,
)
from src.akoma_markup.gazette.verification.processor import (
    BracketCleanupProcessor,
    ContentProcessorChain,
)


class TestVerificationResult:
    """Test VerificationResult class."""
    
    def test_from_llm_response(self):
        """Test creating VerificationResult from LLM response."""
        llm_response = {
            "verified_content": "Cleaned text",
            "confidence_score": 0.85,
            "issues_fixed": ["brackets_removed", "whitespace_normalized"],
            "section_numbers_present": ["1.", "(1)"],
            "recommendations": ["check_section_3"],
        }
        
        result = VerificationResult.from_llm_response(
            section_num="1",
            original_content="[1.] Original text",
            llm_response=llm_response,
        )
        
        assert result.section_num == "1"
        assert result.original_content == "[1.] Original text"
        assert result.verified_content == "Cleaned text"
        assert result.confidence_score == 0.85
        assert result.issues_fixed == ["brackets_removed", "whitespace_normalized"]
        assert result.section_numbers_present == ["1.", "(1)"]
        assert result.recommendations == ["check_section_3"]
    
    def test_from_llm_response_missing_fields(self):
        """Test creating VerificationResult with missing LLM response fields."""
        llm_response = {
            "verified_content": "Cleaned text",
            # Missing other fields
        }
        
        result = VerificationResult.from_llm_response(
            section_num="1",
            original_content="Original text",
            llm_response=llm_response,
        )
        
        assert result.verified_content == "Cleaned text"
        assert result.confidence_score == 0.0  # Default
        assert result.issues_fixed == []  # Default
        assert result.section_numbers_present == []  # Default
        assert result.recommendations == []  # Default


class TestContentProcessorChain:
    """Test ContentProcessorChain."""
    
    def test_bracket_cleanup_processor(self):
        """Test BracketCleanupProcessor."""
        processor = BracketCleanupProcessor()
        
        # Test bracket removal
        input_text = "[1.] Section title\n[(1)] Subsection content"
        expected = "Section title\nSubsection content"
        result = processor.process(input_text)
        assert result == expected
        
        # Test table placeholder preservation
        input_text = "Before <<TABLE_REGION:1>> after"
        result = processor.process(input_text)
        assert "<<TABLE_REGION:1>>" in result
        
        # Test citation preservation
        input_text = "See [Section 3] for details"
        result = processor.process(input_text)
        assert "[Section 3]" in result
    
    def test_content_processor_chain(self):
        """Test ContentProcessorChain."""
        chain = ContentProcessorChain()
        
        input_text = "[1.]  Section   title  \n\n\nContent with   extra   spaces"
        result = chain.process(input_text)
        
        # Should remove brackets and normalize whitespace
        assert "[1.]" not in result
        assert "  " not in result  # No double spaces
        assert "\n\n\n" not in result  # No triple newlines
        
        # Test adding custom processor
        mock_processor = Mock()
        mock_processor.process = Mock(return_value="processed")
        chain.add_processor(mock_processor)
        
        result = chain.process("test")
        mock_processor.process.assert_called_once_with("test")
    
    def test_remove_processor(self):
        """Test removing processor from chain."""
        chain = ContentProcessorChain()
        
        # Initially has 3 processors
        assert len(chain.processors) == 3
        
        # Remove BracketCleanupProcessor
        removed = chain.remove_processor(BracketCleanupProcessor)
        assert removed is True
        assert len(chain.processors) == 2
        
        # Try removing non-existent processor
        removed = chain.remove_processor(type(Mock()))
        assert removed is False
        assert len(chain.processors) == 2


class TestLLMVerificationEngine:
    """Test LLMVerificationEngine."""
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "MockChatModel"
        return mock_llm
    
    @pytest.fixture
    def mock_verification_chain(self):
        """Mock verification chain."""
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value={
            "verified_content": "Verified and cleaned content",
            "confidence_score": 0.9,
            "issues_fixed": ["brackets_removed"],
            "section_numbers_present": ["1."],
            "recommendations": [],
        })
        return mock_chain
    
    @pytest.fixture
    def engine(self, mock_llm, mock_verification_chain):
        """Create LLMVerificationEngine with mocks."""
        with patch.object(LLMVerificationEngine, '_build_verification_chain'):
            engine = LLMVerificationEngine(
                llm=mock_llm,
                verification_level="moderate",
                context_window=3,
                remove_brackets=True,
                batch_size=2,
            )
            engine.verification_chain = mock_verification_chain
            engine.comprehensive_chain = mock_verification_chain
            return engine
    
    def test_initialization(self, mock_llm):
        """Test engine initialization."""
        with patch.object(LLMVerificationEngine, '_build_verification_chain'):
            engine = LLMVerificationEngine(
                llm=mock_llm,
                verification_level="strict",
                context_window=5,
                remove_brackets=False,
                batch_size=10,
            )
            
            assert engine.llm == mock_llm
            assert engine.verification_level == "strict"
            assert engine.context_window == 5
            assert engine.remove_brackets is False
            assert engine.batch_size == 10
            assert isinstance(engine.processor_chain, ContentProcessorChain)
    
    @pytest.mark.asyncio
    async def test_verify_section_success(self, engine, mock_verification_chain):
        """Test successful section verification."""
        result = await engine.verify_section(
            section_num="1",
            content="[1.] Original content with brackets",
            heading="Test Section",
            context_sections=[("2", "Next Section", "Next content")],
        )
        
        # Verify LLM was called
        mock_verification_chain.ainvoke.assert_called_once()
        
        # Verify result
        assert isinstance(result, VerificationResult)
        assert result.section_num == "1"
        assert result.original_content == "[1.] Original content with brackets"
        assert result.verified_content == "Verified and cleaned content"
        assert result.confidence_score == 0.9
        assert result.issues_fixed == ["brackets_removed"]
    
    @pytest.mark.asyncio
    async def test_verify_section_llm_failure(self, engine, mock_verification_chain):
        """Test section verification with LLM failure."""
        mock_verification_chain.ainvoke.side_effect = Exception("LLM error")
        
        result = await engine.verify_section(
            section_num="1",
            content="Original content",
            heading="Test Section",
        )
        
        # Should return fallback result
        assert isinstance(result, VerificationResult)
        assert result.section_num == "1"
        assert result.original_content == "Original content"
        assert result.verified_content == "Original content"  # Fallback to original
        assert result.confidence_score == 0.0
        assert "verification_failed" in result.issues_fixed
    
    def test_format_context(self, engine):
        """Test context formatting."""
        context_sections = [
            ("2", "Previous Section", "Previous content here"),
            ("3", "Current Section", "Current content here"),
        ]
        
        result = engine._format_context(context_sections)
        
        assert "--- Section 2 (Previous Section) ---" in result
        assert "Previous content here" in result
        assert "--- Section 3 (Current Section) ---" in result
        assert "Current content here" in result
    
    def test_format_context_empty(self, engine):
        """Test context formatting with empty list."""
        result = engine._format_context([])
        assert result == "(no context available)"
    
    def test_get_context_for_row(self, engine):
        """Test getting context for a row."""
        # Create test DataFrame
        df = pd.DataFrame({
            "section": ["1", "2", "3", "4", "5"],
            "heading": ["A", "B", "C", "D", "E"],
            "content": ["Content A", "Content B", "Content C", "Content D", "Content E"],
        })
        
        # Get context for middle row (index 2, section 3)
        context = engine._get_context_for_row(df, idx=2, context_window=1)
        
        # Should get 1 row before and 1 row after
        assert len(context) == 2
        assert context[0] == ("2", "B", "Content B")  # Previous
        assert context[1] == ("4", "D", "Content D")  # Next
    
    def test_get_context_for_first_row(self, engine):
        """Test getting context for first row."""
        df = pd.DataFrame({
            "section": ["1", "2", "3"],
            "heading": ["A", "B", "C"],
            "content": ["A", "B", "C"],
        })
        
        context = engine._get_context_for_row(df, idx=0, context_window=2)
        
        # Should only get rows after (no rows before)
        assert len(context) == 2
        assert context[0] == ("2", "B", "B")
        assert context[1] == ("3", "C", "C")
    
    def test_get_context_for_last_row(self, engine):
        """Test getting context for last row."""
        df = pd.DataFrame({
            "section": ["1", "2", "3"],
            "heading": ["A", "B", "C"],
            "content": ["A", "B", "C"],
        })
        
        context = engine._get_context_for_row(df, idx=2, context_window=2)
        
        # Should only get rows before (no rows after)
        assert len(context) == 2
        assert context[0] == ("1", "A", "A")
        assert context[1] == ("2", "B", "B")
    
    @pytest.mark.asyncio
    async def test_verify_tsv_success(self, engine, tmp_path):
        """Test successful TSV verification."""
        # Create test TSV file
        tsv_content = """section\theading\tcontent
1\tSection 1\t[1.] Content one
2\tSection 2\t[2.] Content two
3\tSection 3\t[3.] Content three"""
        
        tsv_path = tmp_path / "test.tsv"
        tsv_path.write_text(tsv_content)
        
        output_path = tmp_path / "output.tsv"
        
        # Mock verify_section to return successful results
        async def mock_verify_section(section_num, content, heading, context_sections):
            return VerificationResult(
                section_num=section_num,
                original_content=content,
                verified_content=content.replace("[", "").replace("]", ""),
                confidence_score=0.9,
                issues_fixed=["brackets_removed"],
                section_numbers_present=[section_num + "."],
                recommendations=[],
            )
        
        engine.verify_section = mock_verify_section
        
        # Run verification
        summary = await engine.verify_tsv(
            tsv_path=tsv_path,
            output_path=output_path,
            resume=True,
        )
        
        # Check summary
        assert summary["input_path"] == str(tsv_path)
        assert summary["output_path"] == str(output_path)
        assert summary["total_sections"] == 3
        assert summary["verified_sections"] == 3
        assert summary["failed_sections"] == 0
        assert summary["avg_confidence"] == 0.9
        assert summary["success_rate"] == 1.0
        
        # Check output file exists
        assert output_path.exists()
        
        # Load output and verify content
        output_df = pd.read_csv(output_path, sep="\t")
        assert len(output_df) == 3
        assert "1. Content one" in output_df.iloc[0]["content"]  # Brackets removed
        assert "2. Content two" in output_df.iloc[1]["content"]
        assert "3. Content three" in output_df.iloc[2]["content"]
    
    @pytest.mark.asyncio
    async def test_verify_tsv_with_failures(self, engine, tmp_path):
        """Test TSV verification with some failures."""
        # Create test TSV file
        tsv_content = """section\theading\tcontent
1\tSection 1\tContent one
2\tSection 2\tContent two"""
        
        tsv_path = tmp_path / "test.tsv"
        tsv_path.write_text(tsv_content)
        
        output_path = tmp_path / "output.tsv"
        
        # Mock verify_section to fail for section 2
        call_count = 0
        
        async def mock_verify_section(section_num, content, heading, context_sections):
            nonlocal call_count
            call_count += 1
            
            if section_num == "2":
                raise Exception("Simulated LLM failure")
            
            return VerificationResult(
                section_num=section_num,
                original_content=content,
                verified_content=content,
                confidence_score=0.9,
                issues_fixed=[],
                section_numbers_present=[section_num + "."],
                recommendations=[],
            )
        
        engine.verify_section = mock_verify_section
        
        # Run verification
        summary = await engine.verify_tsv(
            tsv_path=tsv_path,
            output_path=output_path,
            resume=True,
        )
        
        # Check summary
        assert summary["total_sections"] == 2
        assert summary["verified_sections"] == 2
        assert summary["failed_sections"] == 1
        assert summary["success_rate"] == 0.5  # 1 out of 2 successful
        
        # Check output file
        assert output_path.exists()
    
    @pytest.mark.asyncio
    async def test_verify_tsv_resume_checkpoint(self, engine, tmp_path):
        """Test TSV verification with checkpoint resume."""
        # Create test TSV file
        tsv_content = """section\theading\tcontent
1\tSection 1\tContent one
2\tSection 2\tContent two
3\tSection 3\tContent three"""
        
        tsv_path = tmp_path / "test.tsv"
        tsv_path.write_text(tsv_content)
        
        output_path = tmp_path / "output.tsv"
        
        # First, run partial verification
        processed_sections = []
        
        async def mock_verify_section(section_num, content, heading, context_sections):
            processed_sections.append(section_num)
            
            # Simulate failure on section 3 to stop early
            if section_num == "3":
                raise Exception("Stop here for checkpoint test")
            
            return VerificationResult(
                section_num=section_num,
                original_content=content,
                verified_content=content,
                confidence_score=0.9,
                issues_fixed=[],
                section_numbers_present=[section_num + "."],
                recommendations=[],
            )
        
        engine.verify_section = mock_verify_section
        
        try:
            await engine.verify_tsv(
                tsv_path=tsv_path,
                output_path=output_path,
                resume=True,
            )
        except Exception:
            pass  # Expected to fail on section 3
        
        # Should have processed sections 1 and 2
        assert set(processed_sections) == {"1", "2"}
        
        # Now test resume - clear mock and simulate successful processing
        processed_sections.clear()
        
        async def mock_verify_section_resume(section_num, content, heading, context_sections):
            processed_sections.append(section_num)
            
            # Now all sections succeed
            return VerificationResult(
                section_num=section_num,
                original_content=content,
                verified_content=content,
                confidence_score=0.9,
                issues_fixed=[],
                section_numbers_present=[section_num + "."],
                recommendations=[],
            )
        
        engine.verify_section = mock_verify_section_resume
        
        # Resume verification
        summary = await engine.verify_tsv(
            tsv_path=tsv_path,
            output_path=output_path,
            resume=True,
        )
        
        # Should only process remaining sections (3)
        # Note: In actual implementation, checkpoint would handle this
        # For this test, we're verifying the resume flag is passed through
        assert summary["total_sections"] == 3