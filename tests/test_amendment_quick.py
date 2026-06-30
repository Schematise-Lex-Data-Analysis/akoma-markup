"""Quick validation tests for amendment extraction (no PDF required)."""

import pytest

from akoma_markup.amendment.footnote_linker import (
    _parse_marker_number,
    extract_marker_from_annotation,
    link_footnotes_to_amendments,
    validate_section_linkages,
)
from akoma_markup.amendment.patterns import ExtractedAmendment, FootnoteContext


class TestQuickValidation:
    """Quick tests that don't require PDFs."""

    def test_amendment_creation(self):
        """Test creating an ExtractedAmendment."""
        amdt = ExtractedAmendment(
            amendment_type="substitution",
            act_number="38",
            act_year="1994",
            target_section="10",
            footnote_marker="\u00B9",
            linkage_method="footnote_marker",
            linkage_confidence="high",
        )

        assert amdt.amendment_type == "substitution"
        assert amdt.amendment_act_id == "Act 38 of 1994"
        assert amdt.target_section == "10"
        assert amdt.footnote_marker == "\u00B9"
        assert amdt.linkage_method == "footnote_marker"
        assert amdt.linkage_confidence == "high"

    def test_amendment_to_dict(self):
        """Test converting amendment to dict."""
        amdt = ExtractedAmendment(
            amendment_type="insertion",
            act_number="10",
            act_year="2000",
        )

        data = amdt.to_dict()
        assert data["amendment_type"] == "insertion"
        assert data["act_number"] == "10"
        assert data["act_year"] == "2000"
        assert data["target_section"] is None

    def test_footnote_context_creation(self):
        """Test creating FootnoteContext."""
        ctx = FootnoteContext(
            marker="\u00B9",
            marker_num=1,
            surrounding_text="Some text here",
            suspected_section="5",
            page_num=10,
            line_num=25,
        )

        assert ctx.marker == "\u00B9"
        assert ctx.marker_num == 1
        assert ctx.suspected_section == "5"
        assert ctx.page_num == 10

    def test_full_linkage_pipeline(self):
        """Test the complete linkage pipeline."""
        # Create amendments
        amendments = [
            ExtractedAmendment(
                amendment_type="substitution",
                act_number="38",
                act_year="1994",
                footnote_marker="\u00B9",
            ),
            ExtractedAmendment(
                amendment_type="insertion",
                act_number="10",
                act_year="2000",
                footnote_marker=None,  # No marker
            ),
        ]

        # Create marker contexts
        markers = [
            FootnoteContext(
                marker="\u00B9",
                marker_num=1,
                surrounding_text="text",
                suspected_section="5",
                page_num=1,
                line_num=1,
            ),
        ]

        toc_sections = [{"num": "5", "title": "Section 5"}]
        page_sections = ["5"]

        # Link
        linked = link_footnotes_to_amendments(
            markers, amendments, page_sections, toc_sections
        )

        # First should be linked via marker
        assert linked[0].target_section == "5"
        assert linked[0].linkage_method == "footnote_marker"
        assert linked[0].linkage_confidence == "high"

        # Second should fall back to page inference
        assert linked[1].target_section == "5"
        assert linked[1].linkage_method == "page_inference"
        assert linked[1].linkage_confidence == "medium"

        # Validate
        validated, warnings = validate_section_linkages(linked, toc_sections)
        assert len(validated) == 2
        assert len(warnings) == 0

    def test_all_unicode_superscripts(self):
        """Test all unicode superscript characters."""
        superscripts = ["\u00B9", "\u00B2", "\u00B3", "\u2074", "\u2075",
                        "\u2076", "\u2077", "\u2078", "\u2079"]

        for i, char in enumerate(superscripts, 1):
            assert _parse_marker_number(char) == i

            # Test extraction from annotation
            line = f"{char} Subs. by Act {i} of 2000"
            marker, num = extract_marker_from_annotation(line)
            assert marker == char
            assert num == i

    def test_amendment_types(self):
        """Test different amendment types are handled."""
        types = ["substitution", "insertion", "deletion"]

        for amdt_type in types:
            amdt = ExtractedAmendment(amendment_type=amdt_type)
            assert amdt.amendment_type == amdt_type

    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        # Empty amendments list
        validated, warnings = validate_section_linkages([], [])
        assert len(validated) == 0
        assert len(warnings) == 0

        # No markers
        amendments = [ExtractedAmendment(amendment_type="substitution")]
        linked = link_footnotes_to_amendments([], amendments, [], [])
        assert linked[0].target_section is None
        assert linked[0].linkage_method == "unknown"


class TestSuccessCriteria:
    """Tests to verify success criteria from AMENDMENT_FIX_PLAN."""

    def test_high_linkage_rate_scenario(self):
        """Simulate >95% linkage rate scenario."""
        # Create 100 amendments, 97 have markers
        amendments = []
        markers = []

        for i in range(100):
            if i < 97:
                # These have markers
                marker_char = str((i % 9) + 1)  # 1-9 repeating
                amendments.append(ExtractedAmendment(
                    amendment_type="substitution",
                    footnote_marker=marker_char,
                ))
                markers.append(FootnoteContext(
                    marker=marker_char,
                    marker_num=int(marker_char),
                    surrounding_text="text",
                    suspected_section="10",
                    page_num=1,
                    line_num=i,
                ))
            else:
                # These don't have markers
                amendments.append(ExtractedAmendment(
                    amendment_type="insertion",
                    footnote_marker=None,
                ))

        toc_sections = [{"num": "10", "title": "Section 10"}]
        page_sections = ["10"]

        linked = link_footnotes_to_amendments(
            markers, amendments, page_sections, toc_sections
        )

        # Count linked
        linked_count = sum(1 for a in linked if a.target_section)
        linkage_rate = linked_count / len(linked) * 100

        # With single section on page, all should be linked
        assert linkage_rate >= 95.0, f"Linkage rate {linkage_rate:.1f}% below 95%"

    def test_section_validation(self):
        """Test that linked sections are validated against TOC."""
        amendments = [
            ExtractedAmendment(
                amendment_type="substitution",
                target_section="5",  # Valid
            ),
            ExtractedAmendment(
                amendment_type="insertion",
                target_section="99",  # Invalid - not in TOC
            ),
        ]

        toc_sections = [{"num": "5", "title": "Section 5"}]

        validated, warnings = validate_section_linkages(amendments, toc_sections)

        assert len(warnings) == 1
        assert "99" in warnings[0]

    def test_no_regressions(self):
        """Test that amendments without markers still get processed."""
        amendments = [
            ExtractedAmendment(
                amendment_type="substitution",
                act_number="38",
                act_year="1994",
                target_section="5",
                footnote_marker=None,
            ),
        ]

        # Without footnote markers, should still work
        linked = link_footnotes_to_amendments([], amendments, ["5"], [{"num": "5"}])

        assert len(linked) == 1
        assert linked[0].amendment_type == "substitution"
        assert linked[0].act_number == "38"
