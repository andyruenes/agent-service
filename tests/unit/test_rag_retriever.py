"""Tests del RAG retriever: ahora se ejecuta SIEMPRE (sin router por keywords)."""

from unittest.mock import AsyncMock, patch

import pytest

from app.graph.state import AgentState
from app.nodes.rag_retriever import rag_retriever_node


def _minimal_state(message: str) -> AgentState:
    return {
        "agent_id": "agent-1",
        "session_id": "sess-1",
        "user_id": "",
        "message": message,
        "correlation_id": "",
        "system_prompt": "",
        "chat_history": [],
        "rag_context": [],
        "active_tools": [],
        "response": "",
        "cached": False,
        "error": None,
    }


@pytest.mark.asyncio
class TestRagRetrieverAlwaysRuns:
    async def test_greeting_still_queries_chromadb(self):
        """Antes 'Hi' se saltaba el RAG; ahora SIEMPRE debe consultar."""
        with patch(
            "app.nodes.rag_retriever.query_similar",
            new=AsyncMock(return_value=["chunk relevante"]),
        ) as mock_query:
            out = await rag_retriever_node(_minimal_state("Hi"))

        mock_query.assert_awaited_once()
        assert out["rag_context"] == ["chunk relevante"]

    async def test_any_message_queries_chromadb(self):
        with patch(
            "app.nodes.rag_retriever.query_similar",
            new=AsyncMock(return_value=["doc"]),
        ) as mock_query:
            out = await rag_retriever_node(_minimal_state("¿Cuál es el horario?"))

        mock_query.assert_awaited_once()
        assert out["rag_context"] == ["doc"]

    async def test_empty_knowledge_base_returns_empty_context(self):
        """Sin documentos relevantes → contexto vacío (el paso 3 lo maneja)."""
        with patch(
            "app.nodes.rag_retriever.query_similar",
            new=AsyncMock(return_value=[]),
        ):
            out = await rag_retriever_node(_minimal_state("algo"))

        assert out["rag_context"] == []

    async def test_chromadb_error_degrades_gracefully(self):
        """Un error en ChromaDB no rompe el grafo: retorna contexto vacío."""
        with patch(
            "app.nodes.rag_retriever.query_similar",
            new=AsyncMock(side_effect=RuntimeError("chroma down")),
        ):
            out = await rag_retriever_node(_minimal_state("algo"))

        assert out["rag_context"] == []
