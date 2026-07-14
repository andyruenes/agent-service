# 🤖 agent-service

Motor de IA del chatbot. Ejecuta el grafo LangGraph, realiza RAG con ChromaDB, cachea respuestas en Redis y participa en la SAGA coreografiada vía RabbitMQ.

Para levantar los tres microservicios con un solo Docker Compose (clones, `project.docker-compose.yml` y puertos), consulta el README del repositorio **admin-service** (sección stack completo).

## Arquitectura

```
agent-service
├── Sin BD propia — procesamiento puro
├── Redis → cache respuestas SHA-256 (TTL 24h)
├── ChromaDB → vector store RAG
├── RabbitMQ → SAGA coreografiada
│   ├── consume: msg.received (queue: chat.messages)
│   ├── publica: agent.response (queue: agent.responses)
│   └── publica: agent.failed (compensación)
├── HTTP → admin-service (config, prompt, tools)
└── HTTP → conversation-service (historial)
```

## Flujo del Grafo LangGraph

```
prepare_context → rag_retriever → generator → END
```

1. **prepare_context**: obtiene prompt, tools e historial vía HTTP
2. **rag_retriever**: busca siempre chunks similares en ChromaDB (sin enrutado por keywords)
3. **generator**: construye el prompt completo y llama a GPT-4o

El agente no decide por keywords si usar RAG. Siempre se recupera contexto de la
base de conocimiento y el generador sigue un procedimiento fijo de 3 pasos:

1. Si la información está en el prompt del agente, responde con eso.
2. Si no, usa la información recuperada de la base de conocimiento.
3. Si no está en ninguno de los dos, responde que no dispone de esa información
   (no inventa ni supone datos).

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Healthcheck (ChromaDB + Redis + RabbitMQ) |
| POST | `/agent/documents/index` | Indexa documento en ChromaDB |
| DELETE | `/agent/documents/{doc_id}/vectors` | Elimina vectores de un documento (requiere `agent_id` como query param) |
| DELETE | `/agent/documents/collection/{agent_id}` | Elimina la colección completa del agente (todos sus documentos). Se usa al borrar un agente entero desde admin-service |
| GET | `/agent/documents/collection/{agent_id}/info` | Info de la colección del agente: nombre y total de vectores indexados |
| POST | `/agent/test/query` | Ejecución directa del grafo (sin RabbitMQ) |
| POST | `/agent/test/query/stream` | Versión SSE streaming del test |

## Variables de entorno

Copia la plantilla y ajusta URLs según ejecutes todo en Docker o contra puertos en el host:

```bash
cp .env.example .env
```

Resumen (detalle y valores por defecto en [`app/config.py`](app/config.py)):

- **`OPENAI_API_KEY`** — obligatoria.
- **`OPENAI_BASE_URL`** — opcional; API compatible con OpenAI (incluye `/v1`). Chat y embeddings usan la misma base si está definida.
- **`CHROMA_HOST`**, **`REDIS_URL`**, **`RABBITMQ_URL`** — en el stack unificado desde el host suelen ser `127.0.0.1` con puertos **8003**, **6379**, **5672** y Rabbit con usuario **`app`**.
- **`ADMIN_SERVICE_URL`**, **`CONVERSATION_SERVICE_URL`** — desde el host: **8001** y **8000**; dentro de Compose: `http://admin-api:8000` y `http://conversation-api:8000`.

## Desarrollo

```bash
# Instalar dependencias de desarrollo
pip install -e ".[dev]"

# Instalar pre-commit hooks
pre-commit install

# Lint + format
ruff check . && ruff format .

# Type checking
mypy app/

# Tests
pytest
```