"""Punto de entrada del agent-service — FastAPI + lifespan."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.documents import router as documents_router
from app.api.test_query import router as test_query_router
from app.cache.redis_cache import close_redis, ping_redis
from app.config import get_settings
from app.messaging.consumer import start_consumer, stop_consumer
from app.messaging.publisher import close_publisher, ping_rabbitmq
from app.rag.indexer import ping_chromadb
from app.schemas import HealthResponse

# ── Logging ──
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.DEBUG),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: conecta a RabbitMQ consumer. Shutdown: cierra conexiones."""
    logger.info("🤖 agent-service iniciando...")

    # Startup: iniciar consumer RabbitMQ en background
    consumer_task = asyncio.create_task(start_consumer())

    logger.info("✅ agent-service listo — puerto 8001")
    yield

    # Shutdown
    logger.info("🔻 agent-service cerrando...")
    consumer_task.cancel()
    await stop_consumer()
    await close_publisher()
    await close_redis()
    logger.info("👋 agent-service detenido")


# ── App ──

app = FastAPI(
    title="agent-service",
    description=(
        "Motor de IA del chatbot. Ejecuta LangGraph, RAG con ChromaDB, "
        "cache Redis y participa en la SAGA coreografiada vía RabbitMQ."
    ),
    version="1.0.0",
    root_path="/api/agent",
    lifespan=lifespan,
)

# ── CORS ──
# CORS habilitado para desarrollo local (pruebas sin Nginx).
# En producción, CORS se gestiona en Nginx como API Gateway.
# Este servicio NO es accesible directamente desde el navegador.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──
app.include_router(documents_router)
app.include_router(test_query_router)


# ── Health Check ──


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Healthcheck del servicio",
    description="Verifica conexión a ChromaDB, Redis y RabbitMQ.",
)
async def health_check() -> HealthResponse:
    """GET /health — health check para Docker."""
    chromadb_ok = ping_chromadb()
    redis_ok = await ping_redis()
    rabbitmq_ok = await ping_rabbitmq()

    return HealthResponse(
        status="ok" if all([chromadb_ok, redis_ok, rabbitmq_ok]) else "degraded",
        service="agent-service",
        chromadb="connected" if chromadb_ok else "disconnected",
        redis="connected" if redis_ok else "disconnected",
        rabbitmq="connected" if rabbitmq_ok else "disconnected",
    )
