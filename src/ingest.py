"""PDF extraction and page-aware character chunking."""
import re
from pathlib import Path

import fitz

from .schemas import ChunkMetadata


def clean_text(text: str) -> str:
    """Remove control characters and normalize whitespace."""
    text = "".join(char for char in text if char.isprintable() or char.isspace())
    return re.sub(r"\s+", " ", text).strip()


def chunk_page_text(
    text: str, source: str, page: int, chunk_size: int = 500, chunk_overlap: int = 100
) -> list[ChunkMetadata]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    text = clean_text(text)
    chunks: list[ChunkMetadata] = []
    start = 0
    ordinal = 0
    while start < len(text):
        proposed_end = min(start + chunk_size, len(text))
        end = proposed_end
        if proposed_end < len(text):
            boundary = text.rfind(" ", start, proposed_end + 1)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(ChunkMetadata(
                chunk_id=f"{source}:p{page}:c{ordinal}", source=source, page=page, text=piece
            ))
            ordinal += 1
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def extract_pdf(pdf_path: str | Path, chunk_size: int = 500, chunk_overlap: int = 100) -> list[ChunkMetadata]:
    path = Path(pdf_path)
    chunks: list[ChunkMetadata] = []
    with fitz.open(path) as document:
        for page_number, pdf_page in enumerate(document, start=1):
            text = clean_text(pdf_page.get_text())
            if len(text) < 20:
                continue
            chunks.extend(chunk_page_text(text, path.name, page_number, chunk_size, chunk_overlap))
    return chunks


def ingest_manuals(manuals_dir: str | Path, chunk_size: int = 500, chunk_overlap: int = 100) -> list[ChunkMetadata]:
    chunks: list[ChunkMetadata] = []
    for path in sorted(Path(manuals_dir).glob("*.pdf")):
        chunks.extend(extract_pdf(path, chunk_size, chunk_overlap))
    return chunks
