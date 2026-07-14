"""Procesador de documentos: extracción de texto y chunking."""

import io
import logging

logger = logging.getLogger(__name__)


def extract_text(file_bytes: bytes, file_type: str) -> str:
    """Extrae texto plano de un archivo según su tipo."""
    file_type = file_type.lower().strip(".")

    if file_type == "txt":
        return file_bytes.decode("utf-8", errors="replace")

    if file_type == "md":
        return file_bytes.decode("utf-8", errors="replace")

    if file_type == "pdf":
        return _extract_pdf(file_bytes)

    if file_type == "docx":
        return _extract_docx(file_bytes)

    msg = f"Tipo de archivo no soportado: {file_type}"
    raise ValueError(msg)


def _extract_pdf(file_bytes: bytes) -> str:
    """Extrae texto de un PDF usando PyPDF2."""
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except ImportError:
        logger.warning("PyPDF2 no instalado, intentando con pdfplumber")
        import pdfplumber

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n\n".join(pages)


def _extract_docx(file_bytes: bytes) -> str:
    """Extrae texto de un DOCX usando python-docx."""
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Divide texto en chunks por número aproximado de tokens (palabras).

    Usa tokenización por palabras como aproximación simple.
    chunk_size: número aproximado de tokens por chunk.
    chunk_overlap: tokens de solapamiento entre chunks consecutivos.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - chunk_overlap

    logger.info(
        "Texto dividido en %d chunks (size=%d, overlap=%d)", len(chunks), chunk_size, chunk_overlap
    )
    return chunks
