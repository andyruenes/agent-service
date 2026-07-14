"""Indexador de documentos en ChromaDB con embeddings de OpenAI.

Cada agente tiene su propia colección en ChromaDB.
Nombre de colección: {prefix}_{agent_id}  (ej: "agent_abc123")

Esto permite:
  - Búsquedas más rápidas (el índice solo tiene vectores del agente).
  - Eliminación limpia (borrar un agente = borrar su colección entera).
  - Aislamiento total (un agente nunca ve los documentos de otro).
"""

from __future__ import annotations

import logging
import uuid

import chromadb
from openai import AsyncOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

_chroma_client: chromadb.HttpClient | None = None


# ═══════════════════════════════════════════
#  CONEXIÓN A CHROMADB
# ═══════════════════════════════════════════


def get_chroma_client() -> chromadb.HttpClient:
    """Obtiene o crea el cliente singleton de ChromaDB."""
    global _chroma_client
    if _chroma_client is None:
        settings = get_settings()
        host = settings.chroma_host.replace("http://", "").replace("https://", "")
        parts = host.split(":")
        _chroma_client = chromadb.HttpClient(
            host=parts[0],
            port=int(parts[1]) if len(parts) > 1 else 8000,
        )
    return _chroma_client


def _build_collection_name(agent_id: str) -> str:
    """Construye el nombre de colección para un agente.

    Formato: {prefix}_{agent_id}
    Ejemplo: agent_550e8400-e29b-41d4-a716-446655440000

    ChromaDB requiere que los nombres de colección:
      - Tengan entre 3 y 63 caracteres.
      - Empiecen y terminen con alfanumérico.
      - Solo contengan alfanuméricos, guiones bajos y guiones.
    Los UUIDs cumplen esto naturalmente.
    """
    settings = get_settings()
    name = f"{settings.chroma_collection_prefix}_{agent_id}"
    # Sanitizar: reemplazar caracteres no permitidos
    name = name.replace(" ", "_").lower()
    # Truncar a 63 chars si es necesario (límite de ChromaDB)
    if len(name) > 63:
        name = name[:63]
    return name


def get_collection(agent_id: str) -> chromadb.Collection:
    """Obtiene o crea la colección de un agente específico.

    Si la colección no existe, ChromaDB la crea automáticamente.
    Cada agente tiene su propio índice HNSW con métrica coseno.
    """
    client = get_chroma_client()
    collection_name = _build_collection_name(agent_id)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ═══════════════════════════════════════════
#  EMBEDDINGS
# ═══════════════════════════════════════════


async def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Genera embeddings en batch usando OpenAI text-embedding-3-small."""
    settings = get_settings()
    kwargs: dict[str, str] = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    client = AsyncOpenAI(**kwargs)

    embeddings: list[list[float]] = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await client.embeddings.create(
            input=batch,
            model=settings.openai_embedding_model,
        )
        embeddings.extend([item.embedding for item in response.data])
        logger.debug(
            "Embeddings generados: batch %d/%d",
            i // batch_size + 1,
            (len(texts) + batch_size - 1) // batch_size,
        )

    return embeddings


# ═══════════════════════════════════════════
#  INDEXACIÓN
# ═══════════════════════════════════════════


async def index_chunks(
    chunks: list[str],
    agent_id: str,
    doc_id: str,
    file_name: str = "",
) -> int:
    """Indexa chunks con sus embeddings en la colección del agente.

    Cada chunk se guarda con metadatos doc_id y chunk_index.
    Ya no necesita agent_id en metadatos porque la colección
    es exclusiva del agente.

    Returns:
        Número de chunks indexados.
    """
    if not chunks:
        return 0

    collection_name = _build_collection_name(agent_id)
    logger.info(
        "Indexando %d chunks en colección '%s' para doc_id=%s",
        len(chunks),
        collection_name,
        doc_id,
    )

    for i, chunk in enumerate(chunks):
        logger.debug(
            "📄 Chunk [%d/%d] para doc_id=%s (%s):\n%s",
            i + 1,
            len(chunks),
            doc_id,
            file_name or "sin nombre",
            chunk,
        )

    embeddings = await generate_embeddings(chunks)

    collection = get_collection(agent_id)

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {
            "doc_id": doc_id,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    logger.info(
        "Indexación completada en '%s': %d vectores insertados",
        collection_name,
        len(chunks),
    )
    return len(chunks)


# ═══════════════════════════════════════════
#  BÚSQUEDA RAG
# ═══════════════════════════════════════════


async def query_similar(
    query: str,
    agent_id: str,
    top_k: int = 5,
) -> list[str]:
    """Busca los chunks más similares al query en la colección del agente.

    Ya no necesita filtro where={agent_id} porque la colección
    es exclusiva del agente. La búsqueda es más rápida porque
    el índice solo contiene vectores de este agente.
    """
    embeddings = await generate_embeddings([query])
    query_embedding = embeddings[0]

    collection = get_collection(agent_id)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents"],
    )

    documents = results.get("documents", [[]])[0]
    collection_name = _build_collection_name(agent_id)
    logger.info(
        "RAG query en '%s': %d documentos encontrados",
        collection_name,
        len(documents),
    )

    for i, doc in enumerate(documents):
        logger.debug(
            "🔍 RAG resultado [%d/%d] para query='%s':\n%s",
            i + 1,
            len(documents),
            query[:80],
            doc,
        )

    return documents


# ═══════════════════════════════════════════
#  ELIMINACIÓN
# ═══════════════════════════════════════════


def delete_document_vectors(agent_id: str, doc_id: str) -> int:
    """Elimina todos los vectores de un documento de la colección del agente.

    Returns:
        Número de vectores eliminados.
    """
    collection = get_collection(agent_id)

    existing = collection.get(
        where={"doc_id": doc_id},
        include=[],
    )
    ids_to_delete = existing.get("ids", [])

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    collection_name = _build_collection_name(agent_id)
    logger.info(
        "Eliminados %d vectores de doc_id=%s en colección '%s'",
        len(ids_to_delete),
        doc_id,
        collection_name,
    )
    return len(ids_to_delete)


def delete_agent_collection(agent_id: str) -> bool:
    """Elimina la colección completa de un agente de ChromaDB.

    Se usa cuando se elimina un agente entero desde el admin-service.
    Borra todos los vectores de todos los documentos del agente de golpe.

    Returns:
        True si la colección existía y fue eliminada, False si no existía.
    """
    client = get_chroma_client()
    collection_name = _build_collection_name(agent_id)

    try:
        client.delete_collection(name=collection_name)
        logger.info(
            "Colección '%s' eliminada completamente",
            collection_name,
        )
        return True
    except ValueError:
        logger.info(
            "Colección '%s' no existía, nada que eliminar",
            collection_name,
        )
        return False


def get_agent_collection_info(agent_id: str) -> dict:
    """Retorna información de la colección de un agente.

    Útil para diagnóstico y monitoreo.
    """
    collection = get_collection(agent_id)
    count = collection.count()
    collection_name = _build_collection_name(agent_id)
    return {
        "agent_id": agent_id,
        "collection_name": collection_name,
        "total_vectors": count,
    }


# ═══════════════════════════════════════════
#  HEALTH CHECK
# ═══════════════════════════════════════════


def ping_chromadb() -> bool:
    """Health check de ChromaDB."""
    try:
        client = get_chroma_client()
        client.heartbeat()
        return True
    except Exception:
        return False
