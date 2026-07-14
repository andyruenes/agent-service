"""Nodo Generator: construye el prompt final y llama a GPT-4o."""

import logging
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from app.config import get_settings
from app.graph.state import AgentState

logger = logging.getLogger(__name__)

# Frase EXACTA de derivación a operador humano. DEBE coincidir byte a byte con
# HANDOFF_TO_OPERATOR_PHRASE del conversation-service (app/config.py), que es
# quien la detecta por igualdad exacta para marcar la sesión como derivada.
HANDOFF_TO_OPERATOR_PHRASE = "Derivando a operador humano, enseguida le atenderemos"


def _openai_client() -> AsyncOpenAI:
    settings = get_settings()
    kwargs: dict[str, str] = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return AsyncOpenAI(**kwargs)


def _build_messages(state: AgentState) -> list[dict[str, str]]:
    """Construye la lista de mensajes para la API de OpenAI.

    Estructura:
    1. System prompt (del admin-service)
    2. Contexto RAG (si existe, inyectado en el system)
    3. Historial de chat
    4. Mensaje actual del usuario
    """
    system_parts: list[str] = []

    # Session scope — busts prompt-cache across conversations
    session_id = state.get("session_id", "")
    if session_id:
        system_parts.append(f"[session:{session_id}]")

    # System prompt del agente
    system_prompt = state.get("system_prompt", "")
    if system_prompt:
        system_parts.append(system_prompt)

    # Procedimiento de respuesta — el agente sigue SIEMPRE estos pasos en orden:
    #   1) responder desde el prompt; 2) si no, desde la base de conocimiento;
    #   3) si tampoco está, admitir que no tiene la información (sin inventar).
    system_parts.append(
        "## Cómo debes responder (sigue estos pasos en orden)\n"
    "1. PRIMERO busca la información en la sección 'Base de Conocimiento' de más abajo. "
    "Esa es tu fuente única de verdad para catálogo, precios, zonas, documentación y pagos.\n"
    "2. Si la información NO está en la Base de Conocimiento, verifica si está en tus "
    "instrucciones generales (procedimientos, comportamiento, reglas de seguridad).\n"
    "3. Si la información no está en ninguno de los dos lugares, dile al usuario que no "
    "dispones de esa información. Nunca inventes ni supongas datos."
    )

    # Derivación a un operador humano. Cuando el usuario pide hablar con una
    # persona, el modelo debe responder EXACTAMENTE con la frase de handoff
    # (sin nada más), porque el conversation-service la detecta por igualdad
    # exacta para marcar la sesión como derivada.
    system_parts.append(
        "## Derivación a un operador humano\n"
        "Si el usuario pide explícitamente hablar o ser atendido por una persona, "
        "un humano, un agente, un asesor, un comercial, un representante o el "
        "soporte (por ejemplo: 'quiero hablar con un humano', 'quiero comunicarme "
        "con soporte', 'pásame con un comercial', 'necesito una persona'), "
        "responde ÚNICA Y EXACTAMENTE con esta frase, sin añadir, quitar ni "
        "cambiar absolutamente nada (ni saludos, ni signos, ni texto adicional):\n"
        f'"{HANDOFF_TO_OPERATOR_PHRASE}"'
    )

    # Contexto RAG — la sección se incluye SIEMPRE (con los chunks recuperados
    # o con una nota de que no se halló nada), para que el paso 3 funcione.
    rag_context = state.get("rag_context", [])
    if rag_context:
        context_text = "\n\n---\n\n".join(rag_context)
        system_parts.append(f"## Base de Conocimiento\n{context_text}")
    else:
        system_parts.append(
            "## Base de Conocimiento\nNo se encontraron documentos relevantes para esta consulta."
        )

    # Tools activas (informativo para el modelo)
    active_tools = state.get("active_tools", [])
    if active_tools:
        tools_desc = "\n".join(
            f"- {t.get('nombre', t.get('name', 'tool'))}: "
            f"{t.get('descripcion', t.get('description', ''))}"
            for t in active_tools
        )
        system_parts.append(f"## Tools Disponibles\n{tools_desc}")

    messages: list[dict[str, str]] = []

    # System message
    if system_parts:
        messages.append(
            {
                "role": "system",
                "content": "\n\n".join(system_parts),
            }
        )

    # Chat history
    # El historial puede traer roles que NO son válidos para el LLM (p. ej.
    # "operator" tras una derivación, o "error"). El API de chat completions solo
    # acepta system/user/assistant: un mensaje con role="operator" provoca un 400
    # del proveedor y el agente deja de responder. Mapeamos los turnos del
    # operador humano a "assistant" (para conservar el contexto de lo que se le
    # dijo al usuario durante la derivación) y descartamos roles no soportados.
    role_map = {"user": "user", "assistant": "assistant", "operator": "assistant"}
    chat_history = state.get("chat_history", []) or []
    for msg in chat_history:
        msg = msg or {}
        mapped_role = role_map.get(msg.get("role", "user"))
        content = msg.get("content")
        if mapped_role and content:
            messages.append({"role": mapped_role, "content": content})

    # Mensaje actual del usuario
    messages.append({"role": "user", "content": state["message"]})

    return messages


async def generator_node(state: AgentState) -> dict:
    """Llama a GPT-4o con el contexto completo y retorna la respuesta."""
    settings = get_settings()
    client = _openai_client()

    messages = _build_messages(state)

    try:
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=settings.openai_temperature,
            max_tokens=2048,
        )

        content = response.choices[0].message.content or ""
        logger.info(
            "GPT-4o respondió con %d chars para session=%s",
            len(content),
            state.get("session_id", "unknown"),
        )
        return {"response": content, "error": None}

    except Exception as e:
        logger.exception("Error llamando a OpenAI GPT-4o")
        return {"response": "", "error": str(e)}


async def generator_stream(
    state: AgentState,
    usage_out: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Versión streaming del generador — yield token por token.

    Usado por el endpoint SSE de test y por el grafo de producción.

    Si se pasa ``usage_out``, se rellena con el consumo de tokens reportado por
    el modelo (model_name, input_tokens, output_tokens, cache_tokens) a partir
    del chunk final de uso que emite la API cuando se activa include_usage.
    """
    settings = get_settings()
    client = _openai_client()

    messages = _build_messages(state)

    stream = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=settings.openai_temperature,
        max_tokens=2048,
        stream=True,
        stream_options={"include_usage": True},
    )

    async for chunk in stream:
        # El chunk final de uso llega con choices vacío y usage poblado.
        if chunk.usage is not None and usage_out is not None:
            usage = chunk.usage
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
            usage_out.update(
                {
                    "model_name": settings.openai_model,
                    # input no cacheado: evita cobrar tokens cacheados al precio de input
                    "input_tokens": (usage.prompt_tokens or 0) - cached,
                    "output_tokens": usage.completion_tokens or 0,
                    "cache_tokens": cached,
                }
            )

        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
