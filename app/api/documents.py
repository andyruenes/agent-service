"""Router de documentos: indexación y eliminación de vectores en ChromaDB.

Cada agente tiene su propia colección en ChromaDB.
Los endpoints de este router operan sobre la colección del agente indicado.
"""

import logging

import httpx
from fastapi import APIRouter, Form, HTTPException, Query, UploadFile, status

from app.clients.admin_client import AdminClient
from app.config import get_settings
from app.rag.document_processor import chunk_text, extract_text
from app.rag.indexer import (
    _build_collection_name,
    delete_agent_collection,
    delete_document_vectors,
    get_agent_collection_info,
    index_chunks,
)
from app.schemas import (
    CollectionInfoResponse,
    DeleteCollectionResponse,
    DeleteVectorsResponse,
    IndexDocumentResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent/documents", tags=["documents"])

# Tipos de archivo soportados
_ALLOWED_TYPES = {"pdf", "txt", "docx", "md"}
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/index",
    response_model=IndexDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Indexa un documento en la colección ChromaDB del agente",
    description=(
        "Recibe un archivo via multipart/form-data, extrae el texto, "
        "genera chunks, crea embeddings con OpenAI y los indexa en la "
        "colección exclusiva del agente en ChromaDB."
    ),
)
async def index_document(
    file: UploadFile,
    agent_id: str = Form(...),
    doc_id: str = Form(...),
) -> IndexDocumentResponse:
    """POST /agent/documents/index"""
    # ── Validar tipo de archivo ──
    filename = file.filename or "unknown"
    file_ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if file_ext not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tipo de archivo no soportado: {file_ext}. Permitidos: {_ALLOWED_TYPES}",
        )

    # ── Leer contenido ──
    content = await file.read()
    if len(content) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Archivo demasiado grande: {len(content)} bytes. Máximo: {_MAX_FILE_SIZE}",
        )

    admin_client = AdminClient()

    try:
        # ── Extraer texto ──
        text = extract_text(content, file_ext)
        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No se pudo extraer texto del documento.",
            )

        # ── Chunking ──
        settings = get_settings()
        chunks = chunk_text(
            text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        logger.info(
            "Documento %s: %d chars → %d chunks",
            doc_id,
            len(text),
            len(chunks),
        )

        # ── Indexar en ChromaDB (colección del agente) ──
        chunks_total = await index_chunks(
            chunks=chunks,
            agent_id=agent_id,
            doc_id=doc_id,
            file_name=filename,
        )

        # ── Notificar al admin-service (fila en admin DB; si no existe, solo log) ──
        try:
            await admin_client.update_document_status(
                agent_id=agent_id,
                doc_id=doc_id,
                status="indexado",
                chunks_total=chunks_total,
            )
        except httpx.HTTPStatusError as e:
            logger.warning(
                "Indexación en Chroma OK pero admin-service respondió HTTP %s para doc_id=%s: %s",
                e.response.status_code,
                doc_id,
                e.response.text[:500],
            )
        except Exception as exc:
            logger.warning(
                "Indexación en Chroma OK pero falló notificación al admin (doc_id=%s): %s",
                doc_id,
                exc,
            )

        return IndexDocumentResponse(
            doc_id=doc_id,
            agent_id=agent_id,
            chunks_total=chunks_total,
            collection_name=_build_collection_name(agent_id),
            status="indexado",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error indexando documento %s", doc_id)
        try:
            await admin_client.update_document_status(
                agent_id=agent_id,
                doc_id=doc_id,
                status="error",
                error_mensaje=str(e)[:500],
            )
        except httpx.HTTPStatusError as err:
            logger.warning(
                "No se pudo notificar error al admin (HTTP %s) doc_id=%s: %s",
                err.response.status_code,
                doc_id,
                err.response.text[:300],
            )
        except Exception:
            logger.exception("Error notificando fallo al admin-service")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error indexando documento: {e}",
        ) from e


@router.delete(
    "/{doc_id}/vectors",
    response_model=DeleteVectorsResponse,
    summary="Elimina vectores de un documento de la colección del agente",
    description=(
        "Elimina todos los vectores asociados a un doc_id de la colección "
        "ChromaDB del agente. Requiere agent_id como query param."
    ),
)
async def delete_vectors(
    doc_id: str,
    agent_id: str = Query(..., description="ID del agente dueño del documento"),
) -> DeleteVectorsResponse:
    """DELETE /agent/documents/{doc_id}/vectors?agent_id=xxx"""
    try:
        count = delete_document_vectors(agent_id=agent_id, doc_id=doc_id)
        return DeleteVectorsResponse(
            doc_id=doc_id,
            agent_id=agent_id,
            vectors_deleted=count,
            collection_name=_build_collection_name(agent_id),
            status="eliminado",
        )
    except Exception as e:
        logger.exception("Error eliminando vectores de doc_id=%s", doc_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error eliminando vectores: {e}",
        ) from e


@router.delete(
    "/collection/{agent_id}",
    response_model=DeleteCollectionResponse,
    summary="Elimina la colección completa de un agente",
    description=(
        "Borra toda la colección ChromaDB de un agente, incluyendo "
        "todos los vectores de todos sus documentos. Se usa cuando "
        "se elimina un agente entero desde el admin-service."
    ),
)
async def delete_collection(agent_id: str) -> DeleteCollectionResponse:
    """DELETE /agent/documents/collection/{agent_id}"""
    try:
        collection_name = _build_collection_name(agent_id)
        deleted = delete_agent_collection(agent_id)
        return DeleteCollectionResponse(
            agent_id=agent_id,
            collection_name=collection_name,
            deleted=deleted,
            message=(
                f"Colección '{collection_name}' eliminada"
                if deleted
                else f"Colección '{collection_name}' no existía"
            ),
        )
    except Exception as e:
        logger.exception("Error eliminando colección del agent_id=%s", agent_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error eliminando colección: {e}",
        ) from e


@router.get(
    "/collection/{agent_id}/info",
    response_model=CollectionInfoResponse,
    summary="Información de la colección de un agente",
    description="Retorna el nombre de la colección y el total de vectores indexados.",
)
async def collection_info(agent_id: str) -> CollectionInfoResponse:
    """GET /agent/documents/collection/{agent_id}/info"""
    try:
        info = get_agent_collection_info(agent_id)
        return CollectionInfoResponse(**info)
    except Exception as e:
        logger.exception("Error obteniendo info de colección del agent_id=%s", agent_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo info: {e}",
        ) from e
