"""Nodo RAG Retriever: busca documentos similares en ChromaDB."""

import logging

from app.config import get_settings
from app.graph.state import AgentState
from app.rag.indexer import query_similar

logger = logging.getLogger(__name__)


async def rag_retriever_node(state: AgentState) -> dict:
    """Busca los chunks más relevantes en ChromaDB para el mensaje del usuario.

    Se ejecuta SIEMPRE: para toda consulta se intenta recuperar contexto de la
    base de conocimiento del agente. Si ChromaDB no tiene documentos relevantes,
    retorna contexto vacío sin error (el generador lo maneja en el paso 3).
    """
    settings = get_settings()
    agent_id = state["agent_id"]
    message = state["message"]

    try:
        documents = await query_similar(
            query=message,
            agent_id=agent_id,
            top_k=settings.rag_top_k,
        )
        logger.info(
            "RAG recuperó %d chunks para agent_id=%s",
            len(documents),
            agent_id,
        )
        return {"rag_context": documents}

    except Exception:
        logger.exception("Error en RAG retrieval, continuando sin contexto")
        return {"rag_context": []}
