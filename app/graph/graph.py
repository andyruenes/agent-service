"""StateGraph compilado de LangGraph — flujo principal del agente."""

import logging
import time
from collections.abc import Awaitable, Callable

from langgraph.graph import END, StateGraph

from app.cache.redis_cache import get_cached_response, set_cached_response
from app.clients.admin_client import AdminClient
from app.clients.conversation_client import ConversationClient
from app.graph.state import AgentState
from app.nodes.generator import generator_node, generator_stream
from app.nodes.rag_retriever import rag_retriever_node

logger = logging.getLogger(__name__)


# ── Nodo de preparación: obtiene contexto de otros servicios ──


async def prepare_context_node(state: AgentState) -> dict:
    """Obtiene system_prompt, tools e historial de los otros servicios."""
    agent_id = state["agent_id"]
    user_id = state.get("user_id", "")

    admin_client = AdminClient()
    conv_client = ConversationClient()

    # Llamadas paralelas no bloqueantes
    system_prompt = ""
    active_tools: list = []

    # Preservar el historial que ya venga en el estado (p.ej. el que arma
    # admin-service para WhatsApp y envía vía /agent/test/query). Antes se
    # inicializaba en [] y el return pisaba ese historial, dejando al modelo
    # sin contexto -> trataba cada turno como el primero (bienvenida repetida).
    chat_history: list = state.get("chat_history") or []

    try:
        system_prompt = await admin_client.get_active_prompt(agent_id)
    except Exception:
        logger.warning("No se pudo obtener prompt para agent_id=%s", agent_id)

    try:
        active_tools = await admin_client.get_active_tools(agent_id)
    except Exception:
        logger.warning("No se pudo obtener tools para agent_id=%s", agent_id)

    # Solo consultar conv-service si NO llegó historial en el estado.
    if user_id and not chat_history:
        try:
            chat_history = await conv_client.format_history_for_llm(user_id, limit=20)
        except Exception:
            logger.warning("No se pudo obtener historial para user_id=%s", user_id)

    return {
        "system_prompt": system_prompt,
        "active_tools": active_tools,
        "chat_history": chat_history,
    }


# ── Construir y compilar el grafo ──


def build_graph() -> StateGraph:
    """Construye el StateGraph de LangGraph.

    Flujo:
        prepare_context → rag_retriever → generator → END

    No hay router: el RAG se ejecuta siempre y es el generador quien decide,
    según el procedimiento de 3 pasos, si usa el prompt, la base de
    conocimiento, o responde que no tiene la información.
    """
    graph = StateGraph(AgentState)

    # Agregar nodos
    graph.add_node("prepare_context", prepare_context_node)
    graph.add_node("rag_retriever", rag_retriever_node)
    graph.add_node("generator", generator_node)

    # Definir el flujo
    graph.set_entry_point("prepare_context")
    graph.add_edge("prepare_context", "rag_retriever")
    graph.add_edge("rag_retriever", "generator")
    graph.add_edge("generator", END)

    return graph


# Grafo compilado — singleton
_compiled_graph = build_graph().compile()


async def run_agent_graph(
    agent_id: str,
    session_id: str,
    user_id: str,
    message: str,
    correlation_id: str = "",
    chat_history: list[dict] | None = None,
) -> dict:
    """Ejecuta el grafo completo del agente.

    1. Revisa cache Redis.
    2. Si no hay cache, ejecuta el grafo LangGraph.
    3. Guarda respuesta en cache.

    Returns:
        dict con keys: response, cached, latency_ms, error
    """
    start = time.perf_counter()

    history = chat_history or []
    # La cache de respuestas SOLO es segura en turnos sin contexto previo (FAQ).
    # Con conversación en curso, un mismo texto ("si", "coco") significa cosas
    # distintas según el momento; cachear por mensaje produce respuestas repetidas.
    use_cache = not history

    # ── Paso 1: verificar cache ──
    if use_cache:
        cached_response = await get_cached_response(message, agent_id, user_id)
        if cached_response:
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "response": cached_response,
                "cached": True,
                "latency_ms": round(latency_ms, 2),
                "error": None,
            }

    # ── Paso 2: ejecutar grafo ──
    initial_state: AgentState = {
        "agent_id": agent_id,
        "session_id": session_id,
        "user_id": user_id,
        "message": message,
        "correlation_id": correlation_id,
        "system_prompt": "",
        "chat_history": history,
        "rag_context": [],
        "active_tools": [],
        "response": "",
        "cached": False,
        "error": None,
    }

    try:
        result = await _compiled_graph.ainvoke(initial_state)

        response_text = result.get("response", "")
        error = result.get("error")

        if error:
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "response": "",
                "cached": False,
                "latency_ms": round(latency_ms, 2),
                "error": error,
            }

        # ── Paso 3: guardar en cache (solo turnos sin historial) ──
        if use_cache and response_text:
            await set_cached_response(message, agent_id, response_text, user_id)

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Grafo completado en %.2fms session=%s",
            latency_ms,
            session_id,
        )

        return {
            "response": response_text,
            "cached": False,
            "latency_ms": round(latency_ms, 2),
            "error": None,
        }

    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.exception("Error ejecutando el grafo del agente")
        return {
            "response": "",
            "cached": False,
            "latency_ms": round(latency_ms, 2),
            "error": str(e),
        }


async def run_agent_graph_streaming(
    agent_id: str,
    session_id: str,
    user_id: str,
    message: str,
    correlation_id: str = "",
    on_token: Callable[[str], Awaitable[None]] | None = None,
    chat_history: list[dict] | None = None,
) -> dict:
    """Ejecuta el grafo con streaming de tokens.

    Corre prepare_context → rag_retriever de forma normal, luego hace streaming
    del generator y llama on_token por cada chunk.
    Returns dict con keys: response, cached, latency_ms, error

    La cache se consulta DESPUÉS de prepare_context, una vez que se conoce el
    historial efectivo, y solo se usa cuando no hay conversación en curso.
    """
    start = time.perf_counter()

    # ── Paso 1: construir estado inicial ──
    state: AgentState = {
        "agent_id": agent_id,
        "session_id": session_id,
        "user_id": user_id,
        "message": message,
        "correlation_id": correlation_id,
        "system_prompt": "",
        "chat_history": chat_history or [],
        "rag_context": [],
        "active_tools": [],
        "response": "",
        "cached": False,
        "error": None,
    }

    try:
        # ── Paso 2: preparar contexto (rellena historial desde conv-service) ──
        state = {**state, **await prepare_context_node(state)}

        # ── Paso 3: cache solo en turnos sin historial ──
        use_cache = not state.get("chat_history")
        if use_cache:
            cached_response = await get_cached_response(message, agent_id, user_id)
            if cached_response:
                if on_token:
                    await on_token(cached_response)
                latency_ms = (time.perf_counter() - start) * 1000
                return {
                    "response": cached_response,
                    "cached": True,
                    "latency_ms": round(latency_ms, 2),
                    "error": None,
                }

        # ── Paso 4: RAG ──
        state = {**state, **await rag_retriever_node(state)}

        # ── Paso 5: streaming del generator ──
        full_response = ""
        usage: dict = {}
        async for token in generator_stream(state, usage):
            full_response += token
            if on_token:
                await on_token(token)

        # ── Paso 6: guardar en cache (solo turnos sin historial) ──
        if use_cache and full_response:
            await set_cached_response(message, agent_id, full_response, user_id)

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Streaming completado en %.2fms session=%s",
            latency_ms,
            session_id,
        )
        return {
            "response": full_response,
            "cached": False,
            "latency_ms": round(latency_ms, 2),
            "error": None,
            "model_name": usage.get("model_name"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cache_tokens": usage.get("cache_tokens"),
        }

    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.exception("Error en streaming del agente")
        return {
            "response": "",
            "cached": False,
            "latency_ms": round(latency_ms, 2),
            "error": str(e),
        }
