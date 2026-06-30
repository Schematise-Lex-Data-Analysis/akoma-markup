"""Unit tests for footnote marker linking functionality."""

import pytest

from akoma_markup.amendment.footnote_linker import (
    _parse_marker_number,
    extract_marker_from_annotation,
    extract_markers_with_context,
    link_footnotes_to_amendments,
    validate_section_linkages,
)
from akoma_markup.amendment.patterns import ExtractedAmendment, FootnoteContext


class TestParseMarkerNumber:
    """Test the _parse_marker_number helper function."""

    def test_unicode_superscripts(self):
        """Test parsing unicode superscript characters."""
        assert _parse_marker_number("\u00B9") == 1  # ¹
        assert _parse_marker_number("\u00B2") == 2  # ²
        assert _parse_marker_number("\u00B3") == 3  # ³
        assert _parse_marker_number("\u2074") == 4  # ⁴
        assert _parse_marker_number("\u2075") == 5  # ⁵
        assert _parse_marker_number("\u2076") == 6  # ⁶
        assert _parse_marker_number("\u2077") == 7  # ⁷
        assert _parse_marker_number("\u2078") == 8  # ⁸
        assert _parse_marker_number("\u2079") == 9  # ⁹

    def test_plain_numbers(self):
        """Test parsing plain numeric strings."""
        assert _parse_marker_number("1") == 1
        assert _parse_marker_number("5") == 5
        assert _parse_marker_number("10") == 10

    def test_invalid_markers(self):
        """Test parsing invalid marker strings."""
        assert _parse_marker_number("") == 0
        assert _parse_marker_number("abc") == 0
        assert _parse_marker_number("¹²") == 0  # Multiple chars = invalid


class TestExtractMarkerFromAnnotation:
    """Test extracting markers from amendment annotation lines."""

    def test_unicode_superscript_marker(self):
        """Test extracting unicode superscript markers."""
        line = "\u00B9 Subs. by Act 38 of 1994, s. 2, for certain words"
        marker, num = extract_marker_from_annotation(line)
        assert marker == "\u00B9"
        assert num == 1

    def test_bracketed_marker(self):
        """Test extracting bracketed number markers."""
        line = "[1] Subs. by Act 38 of 1994, s. 2, for certain words"
        marker, num = extract_marker_from_annotation(line)
        assert marker == "1"
        assert num == 1

    def test_plain_number_marker(self):
        """Test extracting plain number markers."""
        line = "1. Subs. by Act 38 of 1994, s. 2, for certain words"
        marker, num = extract_marker_from_annotation(line)
        assert marker == "1"
        assert num == 1

    def test_no_marker(self):
        """Test line without marker."""
        line = "Subs. by Act 38 of 1994, s. 2, for certain words"
        marker, num = extract_marker_from_annotation(line)
        assert marker is None
        assert num == 0

    def test_multi_digit_marker(self):
        """Test extracting multi-digit markers."""
        line = "[12] Ins. by Act 10 of 2000, s. 5"
        marker, num = extract_marker_from_annotation(line)
        assert marker == "12"
        assert num == 12


class TestExtractMarkersWithContext:
    """Test extracting footnote markers with surrounding context."""

    def test_single_marker_in_text(self):
        """Test extracting a single marker from section text."""
        text = """10. Definitions.
In this Act, unless the context otherwise requires.\u00B9
"""
        toc_sections = [{"num": "10", "title": "Definitions"}]
        markers = extract_markers_with_context(text, page_num=1,
                                               toc_sections=toc_sections)

        assert len(markers) == 1
        assert markers[0].marker == "\u00B9"
        assert markers[0].marker_num == 1
        assert markers[0].page_num == 1
        assert markers[0].suspected_section == "10"

    def test_multiple_markers_same_page(self):
        """Test extracting multiple markers from same page."""
        text = """10. Definitions.
Words and expressions defined.\u00B9

11. Application of Act.
This Act applies to all banks.\u00B2
"""
        toc_sections = [
            {"num": "10", "title": "Definitions"},
            {"num": "11", "title": "Application"},
        ]
        markers = extract_markers_with_context(text, page_num=1,
                                               toc_sections=toc_sections)

        assert len(markers) == 2
        marker_nums = {m.marker_num for m in markers}
        assert marker_nums == {1, 2}

    def test_no_markers(self):
        """Test text without any markers."""
        text = """10. Definitions.
In this Act, unless the context otherwise requires.
No footnotes here.
"""
        toc_sections = [{"num": "10", "title": "Definitions"}]
        markers = extract_markers_with_context(text, page_num=1,
                                               toc_sections=toc_sections)

        assert len(markers) == 0

    def test_marker_without_valid_section(self):
        """Test marker where section is not in TOC."""
        text = """Some text.\u00B9 More text here."""
        toc_sections = [{"num": "5", "title": "Other Section"}]
        markers = extract_markers_with_context(text, page_num=1,
                                               toc_sections=toc_sections)

        assert len(markers) == 1
        assert markers[0].marker == "\u00B9"
        assert markers[0].suspected_section is None

    def test_bracketed_markers(self):
        """Test extracting bracketed markers like [1], [2]."""
        text = """10. Definitions.
Words defined.[1]

11. Application.
Various applications.[2]
"""
        toc_sections = [
            {"num": "10", "title": "Definitions"},
            {"num": "11", "title": "Application"},
        ]
        markers = extract_markers_with_context(text, page_num=1,
                                               toc_sections=toc_sections)

        assert len(markers) == 2
        assert any(m.marker == "1" for m in markers)
        assert any(m.marker == "2" for m in markers)


class TestLinkFootnotesToAmendments:
    """Test linking amendments to sections via footnote markers."""

    def test_link_via_footnote_marker(self):
        """Test linking amendment using footnote marker correlation."""
        # Create amendment with footnote marker
        amendment = ExtractedAmendment(
            amendment_type="substitution",
            act_number="38",
            act_year="1994",
            section_number="2",
            footnote_marker="\u00B9",
        )

        # Create marker context pointing to section 10
        markers = [
            FootnoteContext(
                marker="\u00B9",
                marker_num=1,
                surrounding_text="Words.",
                suspected_section="10",
                page_num=1,
                line_num=5,
            )
        ]

        toc_sections = [{"num": "10", "title": "Definitions"}]
        page_sections = ["10"]

        linked = link_footnotes_to_amendments(
            markers, [amendment], page_sections, toc_sections
        )

        assert len(linked) == 1
        assert linked[0].target_section == "10"
        assert linked[0].linkage_method == "footnote_marker"
        assert linked[0].linkage_confidence == "high"

    def test_inline_context_preserved(self):
        """Test that inline context is preserved when valid."""
        amendment = ExtractedAmendment(
            amendment_type="substitution",
            act_number="38",
            act_year="1994",
            target_section="5",
            footnote_marker="\u00B9",
        )

        markers = []
        toc_sections = [{"num": "5", "title": "Existing Section"}]
        page_sections = ["5"]

        linked = link_footnotes_to_amendments(
            markers, [amendment], page_sections, toc_sections
        )

        assert linked[0].target_section == "5"
        assert linked[0].linkage_method == "inline_context"
        assert linked[0].linkage_confidence == "high"

    def test_single_section_page_fallback(self):
        """Test fallback to page inference when single section on page."""
        amendment = ExtractedAmendment(
            amendment_type="insertion",
            act_number="10",
            act_year="2000",
            footnote_marker=None,
        )

        markers = []
        toc_sections = [{"num": "15", "title": "Section 15"}]
        page_sections = ["15"]

        linked = link_footnotes_to_amendments(
            markers, [amendment], page_sections, toc_sections
        )

        assert linked[0].target_section == "15"
        assert linked[0].linkage_method == "page_inference"
        assert linked[0].linkage_confidence == "medium"

    def test_multiple_sections_page_fallback(self):
        """Test fallback when multiple sections on page."""
        amendment = ExtractedAmendment(
            amendment_type="deletion",
            act_number="5",
            act_year="1999",
            footnote_marker=None,
        )

        markers = []
        toc_sections = [
            {"num": "10", "title": "Section 10"},
            {"num": "11", "title": "Section 11"},
        ]
        page_sections = ["10", "11"]

        linked = link_footnotes_to_amendments(
            markers, [amendment], page_sections, toc_sections
        )

        assert linked[0].target_section == "10"  # Picks first
        assert linked[0].linkage_method == "page_inference"
        assert linked[0].linkage_confidence == "low"

    def test_unknown_linkage(self):
        """Test when no section can be determined."""
        amendment = ExtractedAmendment(
            amendment_type="deletion",
            act_number="5",
            act_year="1999",
        )

        markers = []
        toc_sections = []
        page_sections = []

        linked = link_footnotes_to_amendments(
            markers, [amendment], page_sections, toc_sections
        )

        assert linked[0].target_section is None
        assert linked[0].linkage_method == "unknown"
        assert linked[0].linkage_confidence == "none"

    def test_multiple_amendments_different_markers(self):
        """Test linking multiple amendments with different markers."""
        amendments = [
            ExtractedAmendment(
                amendment_type="substitution",
                footnote_marker="\u00B9",
            ),
            ExtractedAmendment(
                amendment_type="insertion",
                footnote_marker="\u00B2",
            ),
        ]

        markers = [
            FootnoteContext(
                marker="\u00B9",
                marker_num=1,
                surrounding_text="text",
                suspected_section="10",
                page_num=1,
                line_num=1,
            ),
            FootnoteContext(
                marker="\u00B2",
                marker_num=2,
                surrounding_text="text",
                suspected_section="11",
                page_num=1,
                line_num=2,
            ),
        ]

        toc_sections = [
            {"num": "10", "title": "Section 10"},
            {"num": "11", "title": "Section 11"},
        ]
        page_sections = ["10", "11"]

        linked = link_footnotes_to_amendments(
            markers, amendments, page_sections, toc_sections
        )

        assert linked[0].target_section == "10"
        assert linked[1].target_section == "11"


class TestValidateSectionLinkages:
    """Test validation of section linkages."""

    def test_all_linked_valid(self):
        """Test when all amendments are properly linked."""
        amendments = [
            ExtractedAmendment(
                amendment_type="substitution",
                target_section="10",
            ),
            ExtractedAmendment(
                amendment_type="insertion",
                target_section="11",
            ),
        ]

        toc_sections = [
            {"num": "10", "title": "Section 10"},
            {"num": "11", "title": "Section 11"},
        ]

        validated, warnings = validate_section_linkages(amendments, toc_sections)

        assert len(validated) == 2
        assert len(warnings) == 0

    def test_unlinked_amendments(self):
        """Test detection of unlinked amendments."""
        amendments = [
            ExtractedAmendment(
                amendment_type="substitution",
                target_section="10",
            ),
            ExtractedAmendment(
                amendment_type="insertion",
                target_section=None,
            ),
        ]

        toc_sections = [{"num": "10", "title": "Section 10"}]

        validated, warnings = validate_section_linkages(amendments, toc_sections)

        assert len(validated) == 2
        # Should have warning about unlinked
        assert any("unlinked" in str(w).lower() or "invalid" in str(w).lower()
                   for w in warnings) or len(warnings) == 0

    def test_invalid_section_reference(self):
        """Test detection of invalid section references."""
        amendments = [
            ExtractedAmendment(
                amendment_type="substitution",
                target_section="99",  # Not in TOC
            ),
        ]

        toc_sections = [{"num": "10", "title": "Section 10"}]

        validated, warnings = validate_section_linkages(amendments, toc_sections)

        assert len(warnings) == 1
        assert "99" in warnings[0]

    def test_empty_amendments(self):
        """Test validation with empty amendments list."""
        amendments = []
        toc_sections = [{"num": "10", "title": "Section 10"}]

        validated, warnings = validate_section_linkages(amendments, toc_sections)

        assert len(validated) == 0
        assert len(warnings) == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_marker_section_mismatch(self):
        """Test when marker points to section not in TOC."""
        amendment = ExtractedAmendment(
            amendment_type="substitution",
            footnote_marker="\u00B9",
        )

        markers = [
            FootnoteContext(
                marker="\u00B9",
                marker_num=1,
                surrounding_text="text",
                suspected_section="99",  # Not in TOC
                page_num=1,
                line_num=1,
            )
        ]

        toc_sections = [{"num": "10", "title": "Section 10"}]
        page_sections = ["10"]

        linked = link_footnotes_to_amendments(
            markers, [amendment], page_sections, toc_sections
        )

        # Falls back to page inference
        assert linked[0].target_section == "10"
        assert linked[0].linkage_method == "page_inference"

    def test_unicode_and_ascii_mixed(self):
        """Test with mix of unicode and ASCII markers."""
        text_with_unicode = "Text.\u00B9 More"
        text_with_bracket = "Text.[1] More"

        toc_sections = [{"num": "1", "title": "Section"}]

        markers_unicode = extract_markers_with_context(
            text_with_unicode, 1, toc_sections
        )
        markers_bracket = extract_markers_with_context(
            text_with_bracket, 1, toc_sections
        )

        assert len(markers_unicode) == 1
        assert markers_unicode[0].marker_num == 1

        assert len(markers_bracket) == 1
        assert markers_bracket[0].marker_num == 1

    def test_large_page_number(self):
        """Test with large page numbers."""
        marker = FootnoteContext(
            marker="\u00B9",
            marker_num=1,
            surrounding_text="text",
            suspected_section="10",
            page_num=999,
            line_num=50,
        )

        assert marker.page_num == 999
        assert marker.line_num == 50
