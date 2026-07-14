"""Cliente HTTP asíncrono al conversation-service para obtener historial."""

import logging
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class ConversationClient:
    """Consulta el historial de mensajes vía HTTP al conversation-service."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.conversation_service_url

    async def get_session_messages(
        self,
        user_id: str,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """GET /sessions/{user_id}/messages — historial de la sesión."""
        url = f"{self.base_url}/sessions/{user_id}/messages"
        params = {"limit": limit}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()["items"]

    async def format_history_for_llm(
        self,
        user_id: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        """Obtiene historial y lo formatea como mensajes para el LLM."""
        messages = await self.get_session_messages(user_id, limit=limit)
        formatted: list[dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Los turnos del operador humano se representan como "assistant" para
            # conservar el contexto; el LLM no acepta el rol "operator".
            if role == "operator":
                role = "assistant"
            if role in ("user", "assistant") and content:
                formatted.append({"role": role, "content": content})
        logger.debug(
            "Historial formateado para user_id=%s: %d mensajes",
            user_id,
            len(formatted),
        )
        return formatted
