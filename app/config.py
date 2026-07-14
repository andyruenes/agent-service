"""Configuración centralizada del agent-service con pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── OpenAI ──
    openai_api_key: str
    # Optional: OpenAI-compatible API (e.g. local llama.cpp / vLLM).
    # Use full base including /v1, e.g. http://127.0.0.1:9100/v1
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    # Temperatura baja: bot transaccional, respuestas deterministas y centradas.
    openai_temperature: float = 0.3

    # ── ChromaDB ──
    chroma_host: str = "http://chromadb:8000"
    chroma_collection_prefix: str = "agent"

    # ── Redis ──
    redis_url: str = "redis://redis:6379/0"
    # Solo se cachean turnos sin historial (FAQ). TTL corto para que cambios de
    # catálogo/prompt no queden pegados durante horas.
    cache_ttl_seconds: int = 3600  # 1h

    # ── RabbitMQ ──
    rabbitmq_url: str = "amqp://app:app@rabbitmq:5672/"
    queue_messages: str = "chat.messages"
    queue_responses: str = "agent.responses"
    exchange_name: str = "chatbot"

    # ── Servicios HTTP internos (nombres de servicio del compose unificado) ──
    admin_service_url: str = "http://admin-api:8000"
    conversation_service_url: str = "http://conversation-api:8000"

    # ── RAG ──
    # Catálogos pequeños y estructurados (tablas de zonas, listas de sabores) se
    # fragmentan mal con chunks chicos. Chunk grande + top_k alto asegura que el
    # documento completo entre en contexto en cada consulta.
    chunk_size: int = 1000
    chunk_overlap: int = 100
    rag_top_k: int = 8

    # ── General ──
    environment: str = "dev"
    log_level: str = "DEBUG"


@lru_cache
def get_settings() -> Settings:
    return Settings()
