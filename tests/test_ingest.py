import fitz
import pytest

from src.ingest import chunk_page_text, clean_text, extract_pdf


@pytest.fixture
def manual_pdf(tmp_path):
    path = tmp_path / "manual.pdf"
    document = fitz.open()
    for page_text in ("First page " * 30, "Second page " * 30):
        page = document.new_page()
        page.insert_text((72, 72), page_text)
    document.save(path)
    document.close()
    return path


def test_cleaning_and_chunk_bounds():
    assert clean_text(" alpha\n\t beta\x00 ") == "alpha beta"
    chunks = chunk_page_text("word " * 100, "manual.pdf", 3, 50, 10)
    assert chunks and all(len(chunk.text) <= 50 for chunk in chunks)
    assert all(chunk.source == "manual.pdf" and chunk.page == 3 for chunk in chunks)
    with pytest.raises(ValueError):
        chunk_page_text("words", "manual.pdf", 1, 10, 10)


def test_pdf_page_metadata(manual_pdf):
    chunks = extract_pdf(manual_pdf, chunk_size=80, chunk_overlap=20)
    assert {chunk.page for chunk in chunks} == {1, 2}
    assert all(chunk.source == "manual.pdf" for chunk in chunks)
