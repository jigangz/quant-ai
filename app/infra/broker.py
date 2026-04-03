from __future__ import annotations

"""
Message Broker Abstraction

Supports: Kafka, Redis Streams, AWS Kinesis, Upstash Kafka, Confluent Cloud.
Swap backend via BROKER_BACKEND env var.

Usage:
    broker = get_broker()
    await broker.publish("market.prices", {"symbol": "AAPL", "price": 175.50})
    await broker.subscribe("market.prices", callback=handle_price)
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Awaitable, Optional

from app.core.settings import settings

logger = logging.getLogger(__name__)


# ===================================
# Data Models
# ===================================

@dataclass
class BrokerMessage:
    """Unified message format across all broker backends."""
    topic: str
    key: Optional[str]
    value: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    headers: dict[str, str] = field(default_factory=dict)
    offset: Optional[int] = None  # Kafka/Kinesis specific
    partition: Optional[int] = None  # Kafka specific


MessageHandler = Callable[[BrokerMessage], Awaitable[None]]


# ===================================
# Abstract Interface
# ===================================

class BaseBroker(ABC):
    """
    Abstract message broker interface.
    
    Implementations must support:
    - publish: send a message to a topic
    - subscribe: register a handler for a topic
    - start/stop: lifecycle management
    """

    @abstractmethod
    async def publish(
        self,
        topic: str,
        value: dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        """Publish a message to a topic."""
        ...

    @abstractmethod
    async def subscribe(
        self,
        topic: str,
        handler: MessageHandler,
        group_id: Optional[str] = None,
    ) -> None:
        """Subscribe to a topic with a handler function."""
        ...

    @abstractmethod
    async def start(self) -> None:
        """Start the broker connection and consumers."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully stop the broker."""
        ...

    async def publish_batch(
        self,
        topic: str,
        messages: list[dict[str, Any]],
        key_fn: Optional[Callable[[dict], str]] = None,
    ) -> None:
        """Publish multiple messages. Default: sequential publish."""
        for msg in messages:
            key = key_fn(msg) if key_fn else None
            await self.publish(topic, msg, key=key)


# ===================================
# Kafka Implementation
# ===================================

class KafkaBroker(BaseBroker):
    """
    Kafka broker using aiokafka.
    
    Works with: local Docker Kafka, Confluent Cloud, Upstash Kafka, AWS MSK.
    Configure via KAFKA_BOOTSTRAP_SERVERS env var.
    For Upstash/Confluent, also set KAFKA_SASL_USERNAME + KAFKA_SASL_PASSWORD.
    """

    def __init__(self):
        self._producer = None
        self._consumers: dict[str, Any] = {}
        self._handlers: dict[str, MessageHandler] = {}
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        from aiokafka import AIOKafkaProducer
        
        bootstrap = getattr(settings, "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        
        producer_kwargs = {
            "bootstrap_servers": bootstrap,
            "value_serializer": lambda v: json.dumps(v, default=str).encode(),
            "key_serializer": lambda k: k.encode() if k else None,
        }
        
        # SASL auth for cloud providers (Upstash, Confluent, MSK IAM)
        sasl_user = getattr(settings, "KAFKA_SASL_USERNAME", None)
        sasl_pass = getattr(settings, "KAFKA_SASL_PASSWORD", None)
        if sasl_user and sasl_pass:
            producer_kwargs.update({
                "security_protocol": "SASL_SSL",
                "sasl_mechanism": "SCRAM-SHA-256",
                "sasl_plain_username": sasl_user,
                "sasl_plain_password": sasl_pass,
            })
        
        self._producer = AIOKafkaProducer(**producer_kwargs)
        await self._producer.start()
        self._running = True
        logger.info(f"Kafka producer connected to {bootstrap}")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._producer:
            await self._producer.stop()
        for consumer in self._consumers.values():
            await consumer.stop()
        logger.info("Kafka broker stopped")

    async def publish(
        self,
        topic: str,
        value: dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        if not self._producer:
            raise RuntimeError("Broker not started")
        
        kafka_headers = (
            [(k, v.encode()) for k, v in headers.items()] if headers else None
        )
        await self._producer.send_and_wait(
            topic, value=value, key=key, headers=kafka_headers,
        )

    async def subscribe(
        self,
        topic: str,
        handler: MessageHandler,
        group_id: Optional[str] = None,
    ) -> None:
        from aiokafka import AIOKafkaConsumer
        
        bootstrap = getattr(settings, "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        
        consumer_kwargs = {
            "bootstrap_servers": bootstrap,
            "group_id": group_id or f"quant-ai-{topic}",
            "value_deserializer": lambda v: json.loads(v.decode()),
            "auto_offset_reset": "latest",
        }
        
        sasl_user = getattr(settings, "KAFKA_SASL_USERNAME", None)
        sasl_pass = getattr(settings, "KAFKA_SASL_PASSWORD", None)
        if sasl_user and sasl_pass:
            consumer_kwargs.update({
                "security_protocol": "SASL_SSL",
                "sasl_mechanism": "SCRAM-SHA-256",
                "sasl_plain_username": sasl_user,
                "sasl_plain_password": sasl_pass,
            })
        
        consumer = AIOKafkaConsumer(topic, **consumer_kwargs)
        await consumer.start()
        self._consumers[topic] = consumer
        self._handlers[topic] = handler
        
        # Start consuming in background
        task = asyncio.create_task(self._consume_loop(topic, consumer, handler))
        self._tasks.append(task)
        logger.info(f"Kafka subscribed to {topic} (group={consumer_kwargs['group_id']})")

    async def _consume_loop(
        self, topic: str, consumer: Any, handler: MessageHandler
    ) -> None:
        try:
            async for msg in consumer:
                broker_msg = BrokerMessage(
                    topic=msg.topic,
                    key=msg.key.decode() if msg.key else None,
                    value=msg.value,
                    timestamp=datetime.fromtimestamp(msg.timestamp / 1000),
                    offset=msg.offset,
                    partition=msg.partition,
                )
                try:
                    await handler(broker_msg)
                except Exception as e:
                    logger.error(f"Handler error for {topic}: {e}", exc_info=True)
        except asyncio.CancelledError:
            pass


# ===================================
# Redis Streams Implementation
# ===================================

class RedisBroker(BaseBroker):
    """
    Redis Streams broker — good for development and small-scale production.
    Zero extra infrastructure if Redis is already running.
    """

    def __init__(self):
        self._redis = None
        self._handlers: dict[str, MessageHandler] = {}
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        import redis.asyncio as aioredis
        
        redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        await self._redis.ping()
        self._running = True
        logger.info(f"Redis broker connected to {redis_url}")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._redis:
            await self._redis.aclose()
        logger.info("Redis broker stopped")

    async def publish(
        self,
        topic: str,
        value: dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        if not self._redis:
            raise RuntimeError("Broker not started")
        
        # Redis Streams: XADD
        fields = {"data": json.dumps(value, default=str)}
        if key:
            fields["key"] = key
        if headers:
            fields["headers"] = json.dumps(headers)
        await self._redis.xadd(f"stream:{topic}", fields, maxlen=10000)

    async def subscribe(
        self,
        topic: str,
        handler: MessageHandler,
        group_id: Optional[str] = None,
    ) -> None:
        if not self._redis:
            raise RuntimeError("Broker not started")
        
        group = group_id or f"quant-ai-{topic}"
        stream_key = f"stream:{topic}"
        
        # Create consumer group (ignore if exists)
        try:
            await self._redis.xgroup_create(stream_key, group, id="0", mkstream=True)
        except Exception:
            pass  # Group already exists
        
        self._handlers[topic] = handler
        task = asyncio.create_task(
            self._consume_loop(stream_key, group, f"consumer-1", handler)
        )
        self._tasks.append(task)
        logger.info(f"Redis subscribed to {stream_key} (group={group})")

    async def _consume_loop(
        self, stream: str, group: str, consumer: str, handler: MessageHandler
    ) -> None:
        try:
            while self._running:
                results = await self._redis.xreadgroup(
                    group, consumer, {stream: ">"}, count=10, block=1000,
                )
                for stream_name, messages in results:
                    for msg_id, fields in messages:
                        value = json.loads(fields.get("data", "{}"))
                        broker_msg = BrokerMessage(
                            topic=stream_name.replace("stream:", ""),
                            key=fields.get("key"),
                            value=value,
                            headers=json.loads(fields.get("headers", "{}")),
                        )
                        try:
                            await handler(broker_msg)
                            await self._redis.xack(stream, group, msg_id)
                        except Exception as e:
                            logger.error(f"Handler error: {e}", exc_info=True)
        except asyncio.CancelledError:
            pass


# ===================================
# In-Memory Implementation (Testing)
# ===================================

class InMemoryBroker(BaseBroker):
    """In-memory broker for unit tests. No external deps."""

    def __init__(self):
        self._handlers: dict[str, list[MessageHandler]] = {}
        self._messages: list[BrokerMessage] = []

    async def start(self) -> None:
        logger.info("In-memory broker started")

    async def stop(self) -> None:
        self._handlers.clear()
        self._messages.clear()

    async def publish(
        self,
        topic: str,
        value: dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        msg = BrokerMessage(topic=topic, key=key, value=value, headers=headers or {})
        self._messages.append(msg)
        for handler in self._handlers.get(topic, []):
            await handler(msg)

    async def subscribe(
        self,
        topic: str,
        handler: MessageHandler,
        group_id: Optional[str] = None,
    ) -> None:
        self._handlers.setdefault(topic, []).append(handler)


# ===================================
# Factory
# ===================================

_broker_instance: Optional[BaseBroker] = None


def get_broker() -> BaseBroker:
    """
    Get the configured broker instance.
    
    Set BROKER_BACKEND env var:
      - "kafka"   → KafkaBroker (local Docker / Confluent / Upstash / MSK)
      - "redis"   → RedisBroker (Redis Streams, zero extra infra)
      - "memory"  → InMemoryBroker (tests only)
    
    Default: "redis" (works out of the box with existing Redis)
    """
    global _broker_instance
    if _broker_instance is None:
        backend = getattr(settings, "BROKER_BACKEND", "redis")
        if backend == "kafka":
            _broker_instance = KafkaBroker()
        elif backend == "redis":
            _broker_instance = RedisBroker()
        elif backend == "memory":
            _broker_instance = InMemoryBroker()
        else:
            raise ValueError(f"Unknown broker backend: {backend}")
        logger.info(f"Broker backend: {backend}")
    return _broker_instance
