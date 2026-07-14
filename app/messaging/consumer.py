"""Consumer de eventos msg.received desde RabbitMQ (SAGA Paso 2).

Escucha la queue chat.messages y ejecuta el grafo del agente.
Publica agent.response o agent.failed según el resultado.
"""

import asyncio
import json
import logging

import aio_pika

from app.config import get_settings
from app.graph.graph import run_agent_graph_streaming
from app.messaging.publisher import (
    publish_agent_failed,
    publish_agent_response,
    publish_agent_token,
)

logger = logging.getLogger(__name__)

_connection: aio_pika.abc.AbstractRobustConnection | None = None


async def _process_message(message: aio_pika.abc.AbstractIncomingMessage) -> None:
    """Procesa un mensaje msg.received de la queue."""
    async with message.process():
        try:
            body = json.loads(message.body.decode())
            logger.info(
                "Consumido msg.received session=%s correlation=%s",
                body.get("session_id"),
                body.get("correlation_id"),
            )

            session_id = body["session_id"]
            user_id = body["user_id"]
            agent_id = body["agent_id"]
            user_message = body["message"]
            correlation_id = body["correlation_id"]

            # ── Ejecutar el grafo del agente con streaming ──
            async def _on_token(token: str) -> None:
                await publish_agent_token(session_id, user_id, agent_id, token, correlation_id)

            result = await run_agent_graph_streaming(
                agent_id=agent_id,
                session_id=session_id,
                user_id=user_id,
                message=user_message,
                correlation_id=correlation_id,
                on_token=_on_token,
            )

            if result.get("error"):
                # ── Compensación: publicar agent.failed ──
                await publish_agent_failed(
                    session_id=session_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    correlation_id=correlation_id,
                    error=result["error"],
                )
            else:
                # ── Éxito: publicar agent.response ──
                await publish_agent_response(
                    session_id=session_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    response=result["response"],
                    correlation_id=correlation_id,
                    cached=result.get("cached", False),
                    model_name=result.get("model_name"),
                    input_tokens=result.get("input_tokens"),
                    output_tokens=result.get("output_tokens"),
                    cache_tokens=result.get("cache_tokens"),
                )

        except Exception as e:
            logger.exception("Error procesando msg.received")
            # Intentar publicar compensación si tenemos los datos
            try:
                body = json.loads(message.body.decode())
                await publish_agent_failed(
                    session_id=body.get("session_id", "unknown"),
                    user_id=body.get("user_id", "unknown"),
                    agent_id=body.get("agent_id", "unknown"),
                    correlation_id=body.get("correlation_id", "unknown"),
                    error=str(e),
                )
            except Exception:
                logger.exception("Error publicando agent.failed de compensación")


async def start_consumer() -> None:
    """Inicia el consumer de RabbitMQ — escucha la queue chat.messages."""
    global _connection
    settings = get_settings()

    logger.info("Conectando consumer a RabbitMQ...")
    _connection = await aio_pika.connect_robust(settings.rabbitmq_url)
    channel = await _connection.channel()

    # Prefetch 1 para procesamiento secuencial por consumidor
    await channel.set_qos(prefetch_count=1)

    # Declarar exchange y queue
    exchange = await channel.declare_exchange(
        settings.exchange_name,
        aio_pika.ExchangeType.TOPIC,
        durable=True,
    )

    queue = await channel.declare_queue(
        settings.queue_messages,
        durable=True,
    )

    # Bind: routing_key msg.received → queue chat.messages
    await queue.bind(exchange, routing_key="msg.received")

    # Empezar a consumir
    await queue.consume(_process_message)

    logger.info(
        "Consumer iniciado — escuchando queue=%s routing_key=msg.received",
        settings.queue_messages,
    )
    # Mantener viva la task del consumer; queue.consume() solo registra callback.
    await asyncio.Event().wait()


async def stop_consumer() -> None:
    """Detiene el consumer y cierra la conexión."""
    global _connection
    if _connection and not _connection.is_closed:
        await _connection.close()
    _connection = None
    logger.info("Consumer detenido")
