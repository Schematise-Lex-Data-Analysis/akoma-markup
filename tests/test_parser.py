import logging

import pytest

from akoma_markup.parser import (
    _find_section_boundary,
    _heading_similarity,
    extract_chapter_ranges,
    extract_section_content,
    filter_sections_by_chapters,
    parse_toc,
    preprocess_pdf_text,
    validate_toc_sections,
)


class TestPreprocessPdfText:
    def test_removes_standalone_page_numbers(self):
        text = "some text\n  42  \nmore text"
        assert "42" not in preprocess_pdf_text(text)

    def test_removes_gazette_header(self):
        text = "THE GAZETTE OF INDIA\nbody text here"
        out = preprocess_pdf_text(text)
        assert "GAZETTE" not in out
        assert "body text here" in out

    def test_removes_extraordinary_header(self):
        text = "EXTRAORDINARY\nactual content"
        assert "EXTRAORDINARY" not in preprocess_pdf_text(text)

    def test_strips_footnote_markers_before_section_numbers(self):
        text = "\n1[10. Something here"
        out = preprocess_pdf_text(text)
        assert "1[10." not in out
        assert "10." in out

    def test_strips_closing_bracket_after_section_number(self):
        text = "10.] Content"
        assert "10." in preprocess_pdf_text(text)
        assert "10.]" not in preprocess_pdf_text(text)

    def test_normalizes_multiple_newlines(self):
        text = "line1\n\n\n\nline2"
        out = preprocess_pdf_text(text)
        assert "\n\n\n" not in out

    def test_strips_whitespace(self):
        assert preprocess_pdf_text("   hello   ") == "hello"


class TestHeadingSimilarity:
    def test_exact_match(self):
        assert _heading_similarity("Arrest", "Arrest") == 1.0

    def test_case_insensitive(self):
        assert _heading_similarity("ARREST", "arrest") == 1.0

    def test_ignores_trailing_punctuation(self):
        assert _heading_similarity("Arrest.", "Arrest") == 1.0

    def test_completely_different(self):
        assert _heading_similarity("Arrest", "Zebra") < 0.5


class TestParseToc:
    def test_extracts_chapters_and_sections(self):
        lines = [
            "THE ACT",
            "",
            "CHAPTER I",
            "PRELIMINARY",
            "1. Short title.",
            "2. Definitions.",
            "CHAPTER II",
            "ARREST",
            "3. When police may arrest.",
            "CHAPTER I",
            "PRELIMINARY",
            "body of section 1 starts here",
        ]
        chapters, sections, toc_end = parse_toc(lines)
        assert len(chapters) == 2
        assert chapters[0]["roman"] == "I"
        assert chapters[0]["heading"] == "PRELIMINARY"
        assert chapters[1]["roman"] == "II"
        assert {s["num"] for s in sections} == {"1", "2", "3"}
        assert toc_end > 0

    def test_handles_alpha_suffix_sections(self):
        lines = [
            "THE ACT",
            "CHAPTER I",
            "PRELIMINARY",
            "1. Title.",
            "1A. Extension.",
            "2. Defs.",
        ]
        _, sections, _ = parse_toc(lines)
        nums = [s["num"] for s in sections]
        assert "1A" in nums

    def test_joins_multiline_headings(self):
        lines = [
            "ACT",
            "CHAPTER I",
            "PART",
            "1. Section heading that",
            "continues on next line.",
            "2. Another.",
        ]
        _, sections, _ = parse_toc(lines)
        first = next(s for s in sections if s["num"] == "1")
        assert "continues on next line" in first["heading"]

    def test_returns_empty_when_no_toc(self):
        chapters, sections, _ = parse_toc(["just some text", "nothing here"])
        assert chapters == []
        assert sections == []

    def test_stops_on_enactment_title_repeat(self):
        # Mirrors real PDFs (e.g. IT Act): TOC ends when the enactment
        # title appears again at the start of the body text.
        lines = [
            "THE EXAMPLE ACT, 2020",
            "ARRANGEMENT OF SECTIONS",
            "CHAPTER I",
            "PRELIMINARY",
            "SECTIONS",
            "1. Short title.",
            "2. Definitions.",
            "CHAPTER II",
            "DIGITAL SIGNATURE",
            "3. Authentication.",
            # padding to push the repeat past i > 10 guard
            "filler", "filler", "filler", "filler", "filler",
            "THE EXAMPLE ACT, 2020",  # body starts here
            "ACT NO. 1 OF 2020",
            "1. Short title, extent...—(1) This Act may be called...",
        ]
        _, sections, _ = parse_toc(lines)
        # Should pick up TOC sections but stop before the body
        nums = {s["num"] for s in sections}
        assert {"1", "2", "3"} <= nums
        # Crucially: should NOT re-add section 1 from the body
        assert sum(1 for s in sections if s["num"] == "1") == 1


class TestValidateTocSections:
    def test_warns_on_single_gap(self, caplog):
        sections = [{"num": "1"}, {"num": "2"}, {"num": "4"}]
        with caplog.at_level(logging.WARNING, logger="akoma_markup.parser"):
            validate_toc_sections(sections)
        assert any("missing section 3" in r.message for r in caplog.records)

    def test_warns_on_range_gap(self, caplog):
        sections = [{"num": "1"}, {"num": "5"}]
        with caplog.at_level(logging.WARNING, logger="akoma_markup.parser"):
            validate_toc_sections(sections)
        assert any("2-4" in r.message for r in caplog.records)

    def test_no_warnings_on_contiguous(self, caplog):
        sections = [{"num": "1"}, {"num": "2"}, {"num": "3"}]
        with caplog.at_level(logging.WARNING, logger="akoma_markup.parser"):
            validate_toc_sections(sections)
        assert not caplog.records


class TestExtractChapterRanges:
    def test_computes_start_and_end(self):
        lines = [
            "CHAPTER I",
            "PRELIMINARY",
            "1. A.",
            "2. B.",
            "3. C.",
            "CHAPTER II",
            "ARREST",
            "4. D.",
            "5. E.",
        ]
        section_names = [{"num": str(i)} for i in range(1, 6)]
        ranges = extract_chapter_ranges(lines, section_names)
        assert len(ranges) == 2
        assert ranges[0] == {"roman": "I", "heading": "PRELIMINARY", "start": 1, "end": 3}
        assert ranges[1] == {"roman": "II", "heading": "ARREST", "start": 4, "end": 5}

    def test_respects_toc_end_line(self):
        lines = [
            "CHAPTER I",
            "PRELIMINARY",
            "1. A.",
            "CHAPTER II",
            "ARREST",
            "2. B.",
            "extra ignored content",
        ]
        ranges = extract_chapter_ranges(lines, [], toc_end_line=2)
        # Only the first chapter should be processed
        assert len(ranges) == 1
        assert ranges[0]["roman"] == "I"


class TestFilterSectionsByChapters:
    def test_assigns_chapter_info(self):
        sections = [{"num": "1", "heading": "A"}, {"num": "5", "heading": "B"}]
        ranges = [
            {"roman": "I", "heading": "P", "start": 1, "end": 3},
            {"roman": "II", "heading": "Q", "start": 4, "end": 7},
        ]
        result = filter_sections_by_chapters(sections, ranges)
        assert result[0]["chapter_roman"] == "I"
        assert result[1]["chapter_roman"] == "II"

    def test_filters_by_selected_chapters(self):
        sections = [{"num": "1", "heading": "A"}, {"num": "5", "heading": "B"}]
        ranges = [
            {"roman": "I", "heading": "P", "start": 1, "end": 3},
            {"roman": "II", "heading": "Q", "start": 4, "end": 7},
        ]
        result = filter_sections_by_chapters(sections, ranges, selected_chapters=["II"])
        assert len(result) == 1
        assert result[0]["num"] == "5"

    def test_falls_back_to_nearest_chapter(self):
        sections = [{"num": "10", "heading": "A"}]
        ranges = [{"roman": "I", "heading": "P", "start": 1, "end": 5}]
        result = filter_sections_by_chapters(sections, ranges)
        # Section 10 falls outside the range but is assigned to nearest (I)
        assert len(result) == 1
        assert result[0]["chapter_roman"] == "I"

    def test_no_chapters_keeps_all(self):
        sections = [{"num": "1", "heading": "A"}]
        result = filter_sections_by_chapters(sections, [])
        assert len(result) == 1


class TestFindSectionBoundary:
    def test_simple_string_match_with_newline(self):
        text = "preamble\nShort title\n— this section..."
        start, end, ctx = _find_section_boundary(text, "1", "Short title")
        assert start is not None
        assert ctx["match_type"] == "simple_string"

    def test_simple_string_match_with_dash(self):
        text = "xxx Short title— the content"
        start, end, ctx = _find_section_boundary(text, "1", "Short title")
        assert start is not None

    def test_simple_number_pattern_fallback(self):
        text = "preamble text here\n5. Something totally unrelated to heading\nmore body"
        start, _, ctx = _find_section_boundary(text, "5", "Completely Different Heading")
        assert start is not None
        assert ctx["match_type"] == "simple_pattern"

    def test_returns_none_when_not_found(self):
        text = "this document has no matching content"
        start, end, ctx = _find_section_boundary(text, "99", "Nothing")
        assert start is None
        assert end is None
        assert ctx is None

    def test_respects_search_start(self):
        text = "Short title\nfirst occurrence\nlater text\nShort title\nsecond"
        first_idx = text.find("Short title")
        start, _, _ = _find_section_boundary(text, "1", "Short title", search_start=first_idx + 1)
        assert start is not None
        assert start > first_idx

    def test_single_line_real_pdf_format(self):
        # Real Indian legislative PDFs put section heading and body on the
        # same line, joined by en-dash: "1. Heading.–(1) body...".
        text = (
            "CHAPTER I\nPRELIMINARY\n"
            "1. Short title, extent, commencement and application.–(1) This Act may be called the IT Act.\n"
            "(2) It shall extend to the whole of India.\n"
            "2. Definitions.–(1) In this Act, unless the context otherwise requires,–\n"
        )
        start, end, ctx = _find_section_boundary(
            text, "1", "Short title, extent, commencement and application"
        )
        assert start is not None
        # Phase 1 simple-string match is the expected path for this format
        assert ctx["match_type"] == "simple_string"


class TestExtractSectionContent:
    def test_extracts_content_between_sections(self):
        text = (
            "\n1. Short title\n— this Act may be called the Test Act.\n"
            "\n2. Definitions\n— in this Act, unless context otherwise requires.\n"
        )
        section_names = [
            {"num": "1", "heading": "Short title"},
            {"num": "2", "heading": "Definitions"},
        ]
        result = extract_section_content(text, section_names)
        assert len(result) == 2
        assert result[0]["num"] == "1"
        assert "Test Act" in result[0]["content"]
        assert result[1]["num"] == "2"
        assert "context otherwise" in result[1]["content"]

    def test_fallback_for_repealed_sections(self):
        text = "\n1. Short title\n— normal content.\n"
        section_names = [
            {"num": "1", "heading": "Short title"},
            {"num": "2", "heading": "Repealed"},
        ]
        result = extract_section_content(text, section_names)
        assert result[1]["content"] == "[Repealed/Omitted]"

    def test_fallback_empty_for_missing_section(self):
        text = "\n1. Short title\n— content\n"
        section_names = [
            {"num": "1", "heading": "Short title"},
            {"num": "99", "heading": "Missing Heading Nowhere"},
        ]
        result = extract_section_content(text, section_names)
        assert result[1]["content"] == ""

    def test_last_section_runs_to_end(self):
        text = "\n1. Short title\n— only section here till the end"
        result = extract_section_content(text, [{"num": "1", "heading": "Short title"}])
        assert "till the end" in result[0]["content"]

    def test_handles_omitted_sections_from_toc(self):
        # Real PDFs list repealed sections as "20. [Omitted.]" in TOC with
        # no body text. Parser should emit the fallback marker.
        text = (
            "19. Recognition of foreign Certifying Authorities.—some body text\n"
            "21. Licence to issue electronic signature Certificates.—more text\n"
        )
        section_names = [
            {"num": "19", "heading": "Recognition of foreign Certifying Authorities"},
            {"num": "20", "heading": "[Omitted.]"},
            {"num": "21", "heading": "Licence to issue electronic signature Certificates"},
        ]
        result = extract_section_content(text, section_names)
        omitted = next(s for s in result if s["num"] == "20")
        assert omitted["content"] == "[Repealed/Omitted]"
        # Flanking sections still get extracted
        assert "some body text" in next(s for s in result if s["num"] == "19")["content"]
