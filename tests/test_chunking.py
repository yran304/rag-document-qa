from app.ingestion import chunk_text, normalize_text


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text(" hello\n\n  world ") == "hello world"


def test_chunking_preserves_source_and_overlap() -> None:
    chunks = chunk_text("one two three four five six seven eight", "guide.txt", chunk_size=14, overlap=4)
    assert len(chunks) >= 2
    assert all(chunk.source == "guide.txt" for chunk in chunks)
    assert chunks[0].id == "guide.txt:0"
    assert "three" in chunks[0].text and "three" in chunks[1].text

