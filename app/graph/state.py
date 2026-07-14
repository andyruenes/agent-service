"""Estado del grafo LangGraph — AgentState TypedDict."""

from typing import TypedDict


class AgentState(TypedDict):
    """Estado compartido entre todos los nodos del grafo.

    Cada nodo lee y escribe campos de este diccionario.
    LangGraph se encarga de pasarlo entre nodos.
    """

    # ── Input ──
    agent_id: str
    session_id: str
    user_id: str
    message: str
    correlation_id: str

    # ── Contexto obtenido durante el flujo ──
    system_prompt: str
    chat_history: list[dict[str, str]]
    rag_context: list[str]
    active_tools: list[dict]

    # ── Output ──
    response: str
    cached: bool
    error: str | None
