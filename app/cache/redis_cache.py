"""Cache de respuestas en Redis. La clave incluye el estado de la conversación
(historial) para no reutilizar respuestas entre turnos con el mismo texto."""

import hashlib
import json
import logging

import redis.asyncio as redis

from app.config import get_settings

logger = logging.getLogger(__name__)

_redis_client: redis.Redis | None = None


async def get_redis() -> redis.Redis:
    """Obtiene o crea la conexión singleton a Redis."""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
    return _redis_client


async def close_redis() -> None:
    """Cierra la conexión a Redis."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None


def _history_fingerprint(history: list[dict] | None) -> str:
    """Huella corta y estable del historial de la conversación.

    Se incorpora a la clave de cache para que dos turnos con el MISMO texto
    pero distinto contexto (p.ej. "si" en momentos diferentes) NO compartan
    respuesta. Sin historial → "0" (turno tipo FAQ, cacheable de forma segura).
    """
    if not history:
        return "0"
    try:
        serialized = json.dumps(history, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        serialized = str(history)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def build_cache_key(
    query: str,
    agent_id: str,
    user_id: str = "",
    history: list[dict] | None = None,
) -> str:
    """Genera clave SHA-256 a partir de query + agent_id + user_id + historial."""
    fp = _history_fingerprint(history)
    raw = f"{user_id}:{agent_id}:{fp}:{query.strip().lower()}"
    return f"cache:{hashlib.sha256(raw.encode()).hexdigest()}"


async def get_cached_response(
    query: str,
    agent_id: str,
    user_id: str = "",
    history: list[dict] | None = None,
) -> str | None:
    """Busca una respuesta cacheada en Redis."""
    client = await get_redis()
    key = build_cache_key(query, agent_id, user_id, history)
    try:
        cached = await client.get(key)
        if cached:
            logger.info("Cache HIT para key=%s", key[:20])
            return json.loads(cached)["response"]
        logger.debug("Cache MISS para key=%s", key[:20])
        return None
    except Exception:
        logger.exception("Error leyendo cache Redis")
        return None


async def set_cached_response(
    query: str,
    agent_id: str,
    response: str,
    user_id: str = "",
    history: list[dict] | None = None,
) -> None:
    """Guarda una respuesta en cache con TTL configurable."""
    settings = get_settings()
    client = await get_redis()
    key = build_cache_key(query, agent_id, user_id, history)
    try:
        payload = json.dumps({"response": response, "agent_id": agent_id})
        await client.set(key, payload, ex=settings.cache_ttl_seconds)
        logger.info("Cache SET key=%s ttl=%ds", key[:20], settings.cache_ttl_seconds)
    except Exception:
        logger.exception("Error escribiendo cache Redis")


async def ping_redis() -> bool:
    """Health check de Redis."""
    try:
        client = await get_redis()
        return await client.ping()
    except Exception:
        return False
