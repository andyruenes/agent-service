"""Router de test: ejecución directa del grafo LangGraph (sin RabbitMQ)."""

import json
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.cache.redis_cache import get_cached_response, set_cached_response
from app.clients.admin_client import AdminClient
from app.graph.graph import run_agent_graph
from app.graph.state import AgentState
from app.nodes.generator import generator_stream
from app.nodes.rag_retriever import rag_retriever_node
from app.schemas import TestQueryRequest, TestQueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent/test", tags=["test"])


@router.post(
    "/query",
    response_model=TestQueryResponse,
    summary="Ejecuta el grafo LangGraph directamente (sin RabbitMQ)",
    description=(
        "Endpoint de prueba que ejecuta el grafo completo de forma síncrona. "
        "Usado por el entorno de prueba del admin panel."
    ),
)
async def test_query(request: TestQueryRequest) -> TestQueryResponse:
    """POST /agent/test/query

    Body: agent_id, session_id, message.
    Devuelve: response, latency_ms.
    """
    result = await run_agent_graph(
        agent_id=request.agent_id,
        session_id=request.session_id,
        user_id="",
        message=request.message,
        correlation_id=str(uuid.uuid4()),
        chat_history=request.history,
    )

    if result.get("error"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en el agente: {result['error']}",
        )

    return TestQueryResponse(
        response=result["response"],
        latency_ms=result["latency_ms"],
        cached=result.get("cached", False),
    )


@router.post(
    "/query/stream",
    summary="Versión streaming del endpoint de prueba via SSE",
    description=(
        "Content-Type: text/event-stream. Emite data: {token} por cada token. "
        "Finaliza con data: [DONE] incluyendo latency_ms."
    ),
)
async def test_query_stream(request: TestQueryRequest) -> StreamingResponse:
    """POST /agent/test/query/stream

    SSE streaming token por token.
    """
    agent_id = request.agent_id
    message = request.message
    # Cache solo en turnos sin historial; con conversación en curso se ejecuta
    # siempre el grafo para no repetir respuestas viejas.
    use_cache = not request.history

    # ── Verificar cache primero (solo si no hay historial) ──
    start = time.perf_counter()
    cached = await get_cached_response(message, agent_id, "", request.history) if use_cache else None
    if cached:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        async def _cached_stream():
            yield f"data: {json.dumps({'token': cached})}\n\n"
            done_cached = {"done": True, "latency_ms": latency_ms, "cached": True}
            yield f"data: {json.dumps(done_cached)}\n\n"

        return StreamingResponse(
            _cached_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Preparar estado para streaming ──
    admin_client = AdminClient()

    system_prompt = ""
    active_tools: list = []

    try:
        system_prompt = await admin_client.get_active_prompt(agent_id)
    except Exception:
        logger.warning("No se pudo obtener prompt para agent_id=%s", agent_id)

    try:
        active_tools = await admin_client.get_active_tools(agent_id)
    except Exception:
        logger.warning("No se pudo obtener tools para agent_id=%s", agent_id)

    # ── RAG retrieval (siempre) ──
    state: AgentState = {
        "agent_id": agent_id,
        "session_id": request.session_id,
        "user_id": "",
        "message": message,
        "correlation_id": str(uuid.uuid4()),
        "system_prompt": system_prompt,
        "chat_history": request.history,
        "rag_context": [],
        "active_tools": active_tools,
        "response": "",
        "cached": False,
        "error": None,
    }

    # RAG
    rag_result = await rag_retriever_node(state)
    state.update(rag_result)

    # ── Streaming response ──
    async def _event_stream():
        full_response = ""
        try:
            async for token in generator_stream(state):
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"

            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            # Guardar en cache (solo turnos sin historial)
            if use_cache and full_response:
                await set_cached_response(message, agent_id, full_response, "", request.history)

            done_live = {"done": True, "latency_ms": latency_ms, "cached": False}
            yield f"data: {json.dumps(done_live)}\n\n"

        except Exception as e:
            logger.exception("Error durante streaming SSE")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
