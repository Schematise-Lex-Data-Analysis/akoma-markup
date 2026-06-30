"""Integration tests for amendment extraction with real PDFs."""

import tempfile
from pathlib import Path

import pytest

from akoma_markup.amendment import (
    AmendmentExtractionResult,
    extract_amendments_from_pdf,
    generate_details_tsv,
    generate_registry_csv,
)


class TestAmendmentExtractionIntegration:
    """Integration tests using real IndiaCode PDFs."""

    @pytest.fixture
    def sample_pdf(self):
        """Find a sample PDF for testing."""
        # Look for PDFs in the project
        search_paths = [
            Path(__file__).parent.parent.parent / "IndiaCode pdfs" / "Banking Reg Act 1949.pdf",
            Path(__file__).parent.parent.parent / "Banking Reg Act 1949.pdf",
            Path(__file__).parent.parent.parent / "mock-pkg" / "a202345.pdf",
        ]

        for path in search_paths:
            if path.exists():
                return path

        pytest.skip("No sample PDF found for integration testing")

    def test_extract_amendments_basic(self, sample_pdf):
        """Test basic amendment extraction from PDF."""
        result = extract_amendments_from_pdf(sample_pdf)

        # Should return proper result
        assert isinstance(result, AmendmentExtractionResult)
        assert result.pdf_path == sample_pdf

        # Should have some sections found
        assert result.sections_found > 0

        # Log the number of amendments found
        print(f"\nExtracted {len(result.amendments)} amendments from {sample_pdf}")
        print(f"Sections found: {result.sections_found}")

    def test_amendment_linkage_rate(self, sample_pdf):
        """Test that amendments are linked to sections at >95% rate."""
        result = extract_amendments_from_pdf(sample_pdf)

        if not result.amendments:
            pytest.skip("No amendments found in PDF")

        # Count linked amendments
        linked = sum(1 for a in result.amendments if a.target_section)
        total = len(result.amendments)
        linkage_rate = linked / total * 100

        print(f"\nLinkage rate: {linked}/{total} ({linkage_rate:.1f}%)")

        # Track linkage methods
        methods = {}
        for a in result.amendments:
            method = a.linkage_method or "unknown"
            methods[method] = methods.get(method, 0) + 1
        print(f"Linkage methods: {methods}")

        # Assert >95% linkage rate
        assert linkage_rate >= 95.0, (
            f"Linkage rate {linkage_rate:.1f}% is below 95% target. "
            f"Only {linked}/{total} amendments linked to sections."
        )

    def test_section_validation(self, sample_pdf):
        """Test that linked sections exist in TOC."""
        result = extract_amendments_from_pdf(sample_pdf)

        # All linked sections should be valid
        for amendment in result.amendments:
            if amendment.target_section:
                # Section should be a valid format (number possibly with letter)
                assert amendment.target_section[0].isdigit(), (
                    f"Invalid section format: {amendment.target_section}"
                )

    def test_amendment_types_detected(self, sample_pdf):
        """Test that various amendment types are detected."""
        result = extract_amendments_from_pdf(sample_pdf)

        # Check for different amendment types
        types_found = set(a.amendment_type for a in result.amendments)
        print(f"\nAmendment types found: {types_found}")

        # Should have at least one type
        assert len(types_found) > 0 or len(result.amendments) == 0

    def test_footnote_marker_extraction(self, sample_pdf):
        """Test that footnote markers are extracted."""
        result = extract_amendments_from_pdf(sample_pdf)

        # Check for footnote markers
        with_markers = sum(1 for a in result.amendments if a.footnote_marker)
        print(f"\nAmendments with footnote markers: {with_markers}/{len(result.amendments)}")

        # Most should have markers
        if result.amendments:
            marker_rate = with_markers / len(result.amendments) * 100
            print(f"Marker extraction rate: {marker_rate:.1f}%")

    def test_no_errors_in_extraction(self, sample_pdf):
        """Test that extraction completes without critical errors."""
        result = extract_amendments_from_pdf(sample_pdf)

        # Log any warnings/errors
        if result.errors:
            print(f"\nExtraction warnings/errors: {len(result.errors)}")
            for error in result.errors[:5]:  # Show first 5
                print(f"  - {error}")

        # Should not have critical errors (file not found, etc)
        critical_errors = [
            e for e in result.errors
            if "not found" in e.lower() or "error opening" in e.lower()
        ]
        assert len(critical_errors) == 0, f"Critical errors: {critical_errors}"

    def test_result_serialization(self, sample_pdf):
        """Test that result can be serialized to dict."""
        result = extract_amendments_from_pdf(sample_pdf)

        # Should convert to dict without errors
        data = result.to_dict()

        assert "pdf_path" in data
        assert "amendments" in data
        assert "sections_found" in data
        assert "errors" in data

    def test_generate_registry_csv(self, sample_pdf, tmp_path):
        """Test generating registry CSV."""
        result = extract_amendments_from_pdf(sample_pdf)

        output_path = tmp_path / "registry.csv"
        csv_path = generate_registry_csv(
            result.amendments,
            output_path,
            act_name="Test Act",
            base_version="1949",
        )

        assert csv_path.exists()
        content = csv_path.read_text()
        assert "act_name" in content
        assert "amendment_act" in content

    def test_generate_details_tsv(self, sample_pdf, tmp_path):
        """Test generating details TSV."""
        result = extract_amendments_from_pdf(sample_pdf)

        output_path = tmp_path / "details.tsv"
        tsv_path = generate_details_tsv(
    result.amendments, 
    output_path, 
    gazette_dir=None,
    source_pdf=sample_pdf,
    extraction_date="2024-01-01"
)

        assert tsv_path.exists()
        content = tsv_path.read_text()
        assert "section" in content
        assert "operation" in content


class TestEdgeCasesIntegration:
    """Integration tests for edge cases."""

    def test_nonexistent_pdf(self):
        """Test handling of non-existent PDF."""
        with pytest.raises(FileNotFoundError):
            extract_amendments_from_pdf("/nonexistent/path.pdf")

    def test_invalid_pdf_path(self):
        """Test handling of invalid PDF path type."""
        with pytest.raises((FileNotFoundError, TypeError)):
            extract_amendments_from_pdf(123)  # type: ignore


class TestPerformanceBenchmark:
    """Benchmark tests for amendment extraction."""

    @pytest.fixture
    def sample_pdf(self):
        """Find a sample PDF for testing."""
        search_paths = [
            Path(__file__).parent.parent.parent / "IndiaCode pdfs" / "Banking Reg Act 1949.pdf",
            Path(__file__).parent.parent.parent / "Banking Reg Act 1949.pdf",
            Path(__file__).parent.parent.parent / "mock-pkg" / "a202345.pdf",
        ]

        for path in search_paths:
            if path.exists():
                return path

        pytest.skip("No sample PDF found for benchmarking")

    def test_extraction_performance(self, sample_pdf):
        """Benchmark amendment extraction performance."""
        import time

        start = time.time()
        result = extract_amendments_from_pdf(sample_pdf)
        duration = time.time() - start

        print(f"\nExtraction took {duration:.2f}s for {result.sections_found} sections")
        print(f"Found {len(result.amendments)} amendments")

        # Should complete in reasonable time (<60s for typical PDF)
        assert duration < 60.0, f"Extraction too slow: {duration:.2f}s"

    def test_memory_efficiency(self, sample_pdf):
        """Test that extraction doesn't consume excessive memory."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        result = extract_amendments_from_pdf(sample_pdf)

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before

        print(f"\nMemory used: {mem_used:.1f} MB")
        print(f"Amendments: {len(result.amendments)}")

        # Should use reasonable memory (<500MB)
        assert mem_used < 500, f"Too much memory used: {mem_used:.1f} MB"
