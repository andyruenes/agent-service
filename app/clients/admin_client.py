"""Cliente HTTP asíncrono al admin-service para obtener configuración del bot."""

import logging
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class AdminClient:
    """Consulta la config del agente vía HTTP al admin-service."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.admin_service_url

    async def get_agent_config(self, agent_id: str) -> dict[str, Any]:
        """GET /admin/agents/{agent_id} — datos completos del agente."""
        url = f"{self.base_url}/admin/agents/{agent_id}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    async def get_active_prompt(self, agent_id: str) -> str:
        """GET /admin/agents/{agent_id}/prompt — prompt activo."""
        url = f"{self.base_url}/admin/agents/{agent_id}/prompt"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            return data.get("content", "")

    async def get_active_tools(self, agent_id: str) -> list[dict[str, Any]]:
        """GET /admin/agents/{agent_id}/tools — tools activas del agente."""
        url = f"{self.base_url}/admin/agents/{agent_id}/tools"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            tools = resp.json()
            return [t for t in tools if t.get("is_active")]

    async def get_use_cases(self, agent_id: str) -> list[dict[str, Any]]:
        """GET /admin/agents/{agent_id}/use-cases — FAQs del agente."""
        url = f"{self.base_url}/admin/agents/{agent_id}/use-cases"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    async def update_document_status(
        self,
        agent_id: str,
        doc_id: str,
        *,
        status: str,
        chunks_total: int | None = None,
        error_mensaje: str | None = None,
    ) -> None:
        """Notifica al admin-service el resultado de indexación.

        Usa el callback interno IndexingResultCallback
        (PATCH .../internal/documents/.../indexing-result), no el PATCH público
        de documentos (solo permite file_name).
        """
        _ = agent_id  # API antigua; el callback interno identifica solo por document_id
        # IndexingResultCallback: status indexed | error
        if status == "indexado":
            mapped = "indexed"
        elif status == "error":
            mapped = "error"
        else:
            mapped = "indexed"

        url = f"{self.base_url}/admin/internal/documents/{doc_id}/indexing-result"
        payload: dict[str, Any] = {"status": mapped}
        if chunks_total is not None:
            payload["chunks_total"] = chunks_total
        if error_mensaje is not None:
            payload["error_message"] = error_mensaje
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.patch(url, json=payload)
            resp.raise_for_status()
        logger.info(
            "Documento %s notificado al admin: status=%s chunks=%s",
            doc_id,
            mapped,
            chunks_total,
        )
