"""Tests unitarios para el procesador de documentos."""

import pytest

from app.rag.document_processor import chunk_text, extract_text


class TestChunkText:
    """Tests de la función chunk_text."""

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty(self):
        assert chunk_text("   \n\n  ") == []

    def test_single_chunk_when_text_shorter_than_size(self):
        text = "palabra " * 10  # 10 palabras
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1

    def test_multiple_chunks_with_overlap(self):
        text = "palabra " * 1000  # 1000 palabras
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1
        # Verificar que todos los chunks tienen contenido
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_chunk_size_respected(self):
        text = "test " * 500
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)
        for chunk in chunks[:-1]:  # Último chunk puede ser más corto
            words = chunk.split()
            assert len(words) <= 100

    def test_overlap_creates_more_chunks(self):
        text = "word " * 200
        no_overlap = chunk_text(text, chunk_size=100, chunk_overlap=0)
        with_overlap = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(with_overlap) > len(no_overlap)


class TestExtractText:
    """Tests de la función extract_text."""

    def test_txt_extraction(self):
        content = b"Hola mundo, este es un texto de prueba."
        result = extract_text(content, "txt")
        assert "Hola mundo" in result

    def test_md_extraction(self):
        content = b"# Title\n\nMarkdown **body**."
        result = extract_text(content, "md")
        assert "Markdown" in result

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="no soportado"):
            extract_text(b"data", "xlsx")
