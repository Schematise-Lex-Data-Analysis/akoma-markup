import json
from pathlib import Path

from akoma_markup.writer import write_markup, write_metadata, write_ocr_text


def test_write_ocr_text_creates_sidecar(tmp_path):
    output = tmp_path / "law_markup.txt"
    result = write_ocr_text("raw ocr content", str(output))
    assert Path(result) == output.with_suffix(".ocr.txt")
    assert Path(result).read_text(encoding="utf-8") == "raw ocr content"


def test_write_ocr_text_handles_unicode(tmp_path):
    output = tmp_path / "doc_markup.txt"
    text = "ह ि न ् द ी — em dash"
    path = write_ocr_text(text, str(output))
    assert Path(path).read_text(encoding="utf-8") == text


def test_write_markup_groups_by_chapter(tmp_path):
    sections = [
        {"num": "1", "markup": "SEC 1. - Title", "chapter_roman": "I", "chapter_heading": "PRELIMINARY"},
        {"num": "2", "markup": "SEC 2. - Defs", "chapter_roman": "I", "chapter_heading": "PRELIMINARY"},
        {"num": "3", "markup": "SEC 3. - Arrests", "chapter_roman": "II", "chapter_heading": "ARREST"},
    ]
    output = tmp_path / "out.txt"
    result = write_markup(sections, str(output))
    content = Path(result).read_text(encoding="utf-8")

    # Chapter I appears once, Chapter II appears once
    assert content.count("CHAPTER I\n") == 1
    assert content.count("CHAPTER II\n") == 1
    assert "PRELIMINARY" in content
    assert "ARREST" in content
    assert "SEC 1. - Title" in content
    assert "SEC 2. - Defs" in content
    assert "SEC 3. - Arrests" in content


def test_write_markup_uses_defaults_for_missing_chapter(tmp_path):
    sections = [{"num": "1", "markup": "SEC 1. - Loose"}]
    output = tmp_path / "out.txt"
    write_markup(sections, str(output))
    content = output.read_text(encoding="utf-8")
    assert "CHAPTER NA" in content
    assert "Unknown" in content


def test_write_markup_creates_parent_dirs(tmp_path):
    output = tmp_path / "nested" / "dir" / "out.txt"
    sections = [{"num": "1", "markup": "SEC 1. - X", "chapter_roman": "I", "chapter_heading": "A"}]
    write_markup(sections, str(output))
    assert output.exists()


def test_write_metadata_basic(tmp_path):
    output = tmp_path / "out.txt"
    sections = [
        {"num": "1", "chapter_roman": "I"},
        {"num": "2", "chapter_roman": "II"},
    ]
    errors = [{"num": "3", "error": "boom"}]
    path = write_metadata(sections, errors, str(output))

    data = json.loads(Path(path).read_text())
    assert data["sections_converted"] == 2
    assert data["errors"] == 1
    assert data["chapters"] == 2
    assert data["ocr_file"].endswith(".ocr.txt")
    assert "conversion_date" in data
    # Optional fields are absent when not provided
    assert "document" not in data
    assert "act_number" not in data
    assert "replaces" not in data


def test_write_metadata_with_optional_fields(tmp_path):
    output = tmp_path / "out.txt"
    path = write_metadata(
        [],
        [],
        str(output),
        document_name="Some Act 2023",
        act_number="42 of 2023",
        replaces="Old Act 1900",
    )
    data = json.loads(Path(path).read_text())
    assert data["document"] == "Some Act 2023"
    assert data["act_number"] == "42 of 2023"
    assert data["replaces"] == "Old Act 1900"


def test_write_metadata_path_suffix(tmp_path):
    output = tmp_path / "markup.txt"
    path = write_metadata([], [], str(output))
    assert Path(path) == output.with_suffix(".meta.json")
