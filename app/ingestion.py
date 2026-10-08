"""Plain-text ingestion and character-based chunking, kept intentionally visible."""
from __future__ import annotations

import re
from pathlib import Path

from app.models import Chunk


def normalize_text(text: str) -> str:
    """Collapse incidental whitespace without changing the document's wording."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, source: str, chunk_size: int, overlap: int) -> list[Chunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")

    clean = normalize_text(text)
    if not clean:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        # Prefer a word boundary so chunks remain readable. The overlap preserves
        # a small amount of context across the boundary.
        if end < len(clean):
            boundary = clean.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(Chunk(id=f"{source}:{index}", text=chunk, source=source, chunk_index=index))
            index += 1
        if end >= len(clean):
            break
        next_start = max(end - overlap, start + 1)
        # A character overlap can otherwise begin in the middle of a word.
        # Move to that word's beginning when it still advances the cursor.
        if next_start < len(clean) and next_start > 0 and clean[next_start - 1] != " ":
            previous_space = clean.rfind(" ", start, next_start)
            if previous_space >= start:
                next_start = previous_space + 1
        start = next_start
    return chunks


def load_text_documents(directory: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    """Load every .txt file in a directory. PDF can be a future extension."""
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.txt")):
        chunks.extend(chunk_text(path.read_text(encoding="utf-8"), path.name, chunk_size, overlap))
    return chunks
