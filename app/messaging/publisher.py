"""Publisher de eventos a RabbitMQ para la SAGA coreografiada."""

import json
import logging

import aio_pika

from app.config import get_settings

logger = logging.getLogger(__name__)

_connection: aio_pika.abc.AbstractRobustConnection | None = None
_channel: aio_pika.abc.AbstractChannel | None = None


async def get_channel() -> aio_pika.abc.AbstractChannel:
    """Obtiene o crea la conexión y canal a RabbitMQ."""
    global _connection, _channel
    if _connection is None or _connection.is_closed:
        settings = get_settings()
        _connection = await aio_pika.connect_robust(settings.rabbitmq_url)
        _channel = await _connection.channel()
        # Declarar exchange tipo topic
        await _channel.declare_exchange(
            settings.exchange_name,
            aio_pika.ExchangeType.TOPIC,
            durable=True,
        )
    return _channel  # type: ignore[return-value]


async def publish_agent_token(
    session_id: str,
    user_id: str,
    agent_id: str,
    token: str,
    correlation_id: str,
) -> None:
    """Publica un chunk de token al stream (NON_PERSISTENT, alta frecuencia)."""
    settings = get_settings()
    channel = await get_channel()
    exchange = await channel.get_exchange(settings.exchange_name)

    payload = {
        "event": "agent.token",
        "session_id": session_id,
        "user_id": user_id,
        "agent_id": agent_id,
        "token": token,
        "correlation_id": correlation_id,
    }

    message = aio_pika.Message(
        body=json.dumps(payload).encode(),
        content_type="application/json",
        delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT,
    )

    await exchange.publish(message, routing_key="agent.token")


async def publish_agent_response(
    session_id: str,
    user_id: str,
    agent_id: str,
    response: str,
    correlation_id: str,
    *,
    cached: bool = False,
    model_name: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_tokens: int | None = None,
) -> None:
    """Publica evento agent.response en RabbitMQ (SAGA Paso 2 → 3)."""
    settings = get_settings()
    channel = await get_channel()
    exchange = await channel.get_exchange(settings.exchange_name)

    payload = {
        "event": "agent.response",
        "session_id": session_id,
        "user_id": user_id,
        "agent_id": agent_id,
        "response": response,
        "correlation_id": correlation_id,
        "cached": cached,
        "model_name": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_tokens": cache_tokens,
    }

    message = aio_pika.Message(
        body=json.dumps(payload).encode(),
        content_type="application/json",
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
    )

    await exchange.publish(
        message,
        routing_key="agent.response",
    )

    logger.info(
        "Publicado agent.response session=%s correlation=%s cached=%s",
        session_id,
        correlation_id,
        cached,
    )


async def publish_agent_failed(
    session_id: str,
    user_id: str,
    agent_id: str,
    correlation_id: str,
    error: str,
) -> None:
    """Publica evento agent.failed en RabbitMQ (compensación SAGA)."""
    settings = get_settings()
    channel = await get_channel()
    exchange = await channel.get_exchange(settings.exchange_name)

    payload = {
        "event": "agent.failed",
        "session_id": session_id,
        "user_id": user_id,
        "agent_id": agent_id,
        "correlation_id": correlation_id,
        "error": error,
    }

    message = aio_pika.Message(
        body=json.dumps(payload).encode(),
        content_type="application/json",
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
    )

    await exchange.publish(
        message,
        routing_key="agent.failed",
    )

    logger.warning(
        "Publicado agent.failed session=%s error=%s",
        session_id,
        error[:100],
    )


async def close_publisher() -> None:
    """Cierra la conexión del publisher a RabbitMQ."""
    global _connection, _channel
    if _connection and not _connection.is_closed:
        await _connection.close()
    _connection = None
    _channel = None


async def ping_rabbitmq() -> bool:
    """Health check de RabbitMQ."""
    try:
        channel = await get_channel()
        return not channel.is_closed
    except Exception:
        return False
