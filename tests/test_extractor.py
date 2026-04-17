from unittest.mock import MagicMock, patch

from akoma_markup.extractor import extract_pdf_text


def _make_page(text):
    page = MagicMock()
    page.extract_text.return_value = text
    return page


def _make_pdf_context(pages):
    pdf = MagicMock()
    pdf.pages = pages
    ctx = MagicMock()
    ctx.__enter__.return_value = pdf
    ctx.__exit__.return_value = False
    return ctx


def test_extract_pdf_text_joins_pages():
    pages = [_make_page("page 1 text"), _make_page("page 2 text")]
    with patch("akoma_markup.extractor.pdfplumber.open", return_value=_make_pdf_context(pages)) as mock_open:
        result = extract_pdf_text("/tmp/fake.pdf")
    mock_open.assert_called_once_with("/tmp/fake.pdf")
    assert result == "page 1 text\npage 2 text"


def test_extract_pdf_text_skips_empty_pages():
    pages = [_make_page("first"), _make_page(None), _make_page(""), _make_page("last")]
    with patch("akoma_markup.extractor.pdfplumber.open", return_value=_make_pdf_context(pages)):
        result = extract_pdf_text("/tmp/fake.pdf")
    assert result == "first\nlast"


def test_extract_pdf_text_empty_pdf():
    with patch("akoma_markup.extractor.pdfplumber.open", return_value=_make_pdf_context([])):
        result = extract_pdf_text("/tmp/empty.pdf")
    assert result == ""
