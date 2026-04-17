from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from akoma_markup import convert


def test_convert_raises_for_missing_pdf(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert(str(tmp_path / "missing.pdf"), llm_config={"provider": "anthropic"})


def test_convert_pipeline_happy_path(tmp_path):
    pdf = tmp_path / "act.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    raw_text = "\n".join(
        [
            "THE ACT",
            "",
            "CHAPTER I",
            "PRELIMINARY",
            "1. Short title.",
            "2. Definitions.",
            "CHAPTER I",
            "PRELIMINARY",
            "body starts",
        ]
    )

    with patch("akoma_markup.build_llm") as mock_build_llm, \
         patch("akoma_markup.extract_pdf_text", return_value=raw_text), \
         patch("akoma_markup.build_chain") as mock_build_chain, \
         patch("akoma_markup.process_all_sections") as mock_process, \
         patch("akoma_markup.write_ocr_text") as mock_ocr, \
         patch("akoma_markup.write_markup") as mock_markup, \
         patch("akoma_markup.write_metadata") as mock_meta:

        mock_build_llm.return_value = MagicMock()
        mock_build_chain.return_value = MagicMock()
        mock_process.return_value = (
            [{"num": "1", "markup": "SEC 1. - Title"}],
            [],
        )
        mock_ocr.return_value = str(tmp_path / "act_markup.ocr.txt")
        mock_markup.return_value = str(tmp_path / "act_markup.txt")
        mock_meta.return_value = str(tmp_path / "act_markup.meta.json")

        result = convert(
            str(pdf),
            llm_config={"provider": "anthropic", "api_key": "k"},
            document_name="Act 2023",
            act_number="1 of 2023",
            replaces="Old Act",
        )

    assert result == str(tmp_path / "act_markup.txt")
    mock_build_llm.assert_called_once_with({"provider": "anthropic", "api_key": "k"})
    mock_build_chain.assert_called_once()
    assert mock_build_chain.call_args.kwargs["document_name"] == "Act 2023"

    # Metadata should receive document fields
    meta_kwargs = mock_meta.call_args.kwargs
    assert meta_kwargs["document_name"] == "Act 2023"
    assert meta_kwargs["act_number"] == "1 of 2023"
    assert meta_kwargs["replaces"] == "Old Act"


def test_convert_defaults_output_and_document_name(tmp_path):
    pdf = tmp_path / "mydoc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    with patch("akoma_markup.build_llm"), \
         patch("akoma_markup.extract_pdf_text", return_value="anything"), \
         patch("akoma_markup.parse_toc", return_value=([], [], 0)), \
         patch("akoma_markup.extract_chapter_ranges", return_value=[]), \
         patch("akoma_markup.extract_section_content", return_value=[]), \
         patch("akoma_markup.filter_sections_by_chapters", return_value=[]), \
         patch("akoma_markup.build_chain"), \
         patch("akoma_markup.process_all_sections", return_value=([], [])), \
         patch("akoma_markup.write_ocr_text", return_value=""), \
         patch("akoma_markup.write_markup") as mock_markup, \
         patch("akoma_markup.write_metadata") as mock_meta:

        mock_markup.return_value = str(tmp_path / "mydoc_markup.txt")
        mock_meta.return_value = str(tmp_path / "mydoc_markup.meta.json")

        convert(str(pdf), llm_config={"provider": "anthropic", "api_key": "k"})

    # The output path passed to write_markup should default to <stem>_markup.txt
    output_arg = mock_markup.call_args.args[1]
    assert Path(output_arg).name == "mydoc_markup.txt"
    # Document name defaults to PDF stem
    assert mock_meta.call_args.kwargs["document_name"] == "mydoc"


def test_convert_deduplicates_sections(tmp_path):
    pdf = tmp_path / "dup.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    with patch("akoma_markup.build_llm"), \
         patch("akoma_markup.extract_pdf_text", return_value="anything"), \
         patch("akoma_markup.parse_toc", return_value=([], [], 0)), \
         patch(
             "akoma_markup.extract_section_content",
             return_value=[
                 {"num": "1", "heading": "A", "content": "x"},
                 {"num": "1", "heading": "A", "content": "x"},
                 {"num": "2", "heading": "B", "content": "y"},
             ],
         ), \
         patch("akoma_markup.extract_chapter_ranges", return_value=[]), \
         patch(
             "akoma_markup.filter_sections_by_chapters",
             side_effect=lambda sections, ranges: sections,
         ), \
         patch("akoma_markup.build_chain"), \
         patch("akoma_markup.process_all_sections") as mock_process, \
         patch("akoma_markup.write_ocr_text", return_value=""), \
         patch("akoma_markup.write_markup", return_value=""), \
         patch("akoma_markup.write_metadata", return_value=""):

        mock_process.return_value = ([], [])
        convert(str(pdf), llm_config={"provider": "anthropic", "api_key": "k"})

    # process_all_sections should receive deduplicated sections (2 unique, not 3)
    passed_sections = mock_process.call_args.args[1]
    assert len(passed_sections) == 2
    assert {s["num"] for s in passed_sections} == {"1", "2"}
