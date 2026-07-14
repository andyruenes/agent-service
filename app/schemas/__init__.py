"""Esquemas Pydantic v2 para request/response del agent-service."""

from pydantic import BaseModel, Field

# ── Documents ──


class IndexDocumentResponse(BaseModel):
    doc_id: str
    agent_id: str
    chunks_total: int
    collection_name: str
    status: str = "indexado"


class DeleteVectorsResponse(BaseModel):
    doc_id: str
    agent_id: str
    vectors_deleted: int
    collection_name: str
    status: str = "eliminado"


class DeleteCollectionResponse(BaseModel):
    agent_id: str
    collection_name: str
    deleted: bool
    message: str


class CollectionInfoResponse(BaseModel):
    agent_id: str
    collection_name: str
    total_vectors: int


# ── Test Query ──


class TestQueryRequest(BaseModel):
    agent_id: str
    session_id: str
    message: str = Field(..., min_length=1)
    history: list[dict] = Field(default_factory=list)


class TestQueryResponse(BaseModel):
    response: str
    latency_ms: float
    cached: bool = False


# ── SAGA Events ──


class MessageReceivedEvent(BaseModel):
    """Payload del evento msg.received consumido desde RabbitMQ."""

    session_id: str
    user_id: str
    agent_id: str
    message: str
    correlation_id: str


class AgentResponseEvent(BaseModel):
    """Payload del evento agent.response publicado a RabbitMQ."""

    session_id: str
    user_id: str
    agent_id: str
    response: str
    correlation_id: str
    cached: bool = False
    model_name: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_tokens: int | None = None


class AgentFailedEvent(BaseModel):
    """Payload del evento agent.failed (compensacion SAGA)."""

    session_id: str
    user_id: str
    agent_id: str
    correlation_id: str
    error: str


# ── Health ──


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "agent-service"
    chromadb: str = "connected"
    redis: str = "connected"
    rabbitmq: str = "connected"
