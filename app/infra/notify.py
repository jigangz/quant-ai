"""
Notification Abstraction (SNS / Redis Pub-Sub / Webhook)

Supports: AWS SNS, Redis Pub/Sub, Webhook POST, In-Memory.
Swap backend via NOTIFY_BACKEND env var.

Usage:
    notifier = get_notifier()
    await notifier.publish("alerts", {
        "type": "price_breakout",
        "symbol": "AAPL",
        "message": "AAPL broke above $180 resistance"
    })
    await notifier.subscribe("alerts", handler=handle_alert)
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
class Notification:
    topic: str
    payload: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    subject: Optional[str] = None


NotificationHandler = Callable[[Notification], Awaitable[None]]


# ===================================
# Abstract Interface
# ===================================

class BaseNotifier(ABC):
    """Abstract notification interface with fan-out support."""

    @abstractmethod
    async def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        subject: Optional[str] = None,
    ) -> None:
        """Publish a notification to a topic (fans out to all subscribers)."""
        ...

    @abstractmethod
    async def subscribe(
        self,
        topic: str,
        handler: NotificationHandler,
    ) -> None:
        """Subscribe to a notification topic."""
        ...

    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...


# ===================================
# SNS Implementation
# ===================================

class SNSNotifier(BaseNotifier):
    """
    AWS SNS for notification fan-out.
    
    Works with: real AWS SNS, LocalStack (local dev).
    Set SNS_ENDPOINT_URL=http://localhost:4566 for LocalStack.
    
    Fan-out: one publish → multiple subscribers (Email, SQS, Lambda, HTTP).
    """

    def __init__(self):
        self._client = None
        self._topic_arns: dict[str, str] = {}

    async def _get_client(self):
        if self._client is None:
            import aioboto3
            session = aioboto3.Session()
            endpoint = getattr(settings, "SNS_ENDPOINT_URL", None)
            kwargs = {"service_name": "sns", "region_name": getattr(settings, "AWS_REGION", "us-east-1")}
            if endpoint:
                kwargs["endpoint_url"] = endpoint
            self._client = await session.client(**kwargs).__aenter__()
        return self._client

    async def _get_or_create_topic(self, topic: str) -> str:
        if topic in self._topic_arns:
            return self._topic_arns[topic]
        client = await self._get_client()
        resp = await client.create_topic(Name=f"quant-ai-{topic}")
        arn = resp["TopicArn"]
        self._topic_arns[topic] = arn
        return arn

    async def start(self) -> None:
        await self._get_client()
        logger.info("SNS notifier started")

    async def stop(self) -> None:
        if self._client:
            await self._client.__aexit__(None, None, None)

    async def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        subject: Optional[str] = None,
    ) -> None:
        client = await self._get_client()
        topic_arn = await self._get_or_create_topic(topic)
        await client.publish(
            TopicArn=topic_arn,
            Message=json.dumps(payload, default=str),
            Subject=subject or f"quant-ai:{topic}",
        )

    async def subscribe(
        self,
        topic: str,
        handler: NotificationHandler,
    ) -> None:
        # SNS subscriptions are typically configured via AWS console/Terraform
        # (Email, SQS queue, Lambda, HTTP endpoint)
        # For in-process handlers, use Redis Pub/Sub instead
        logger.warning(
            "SNS subscribe is for AWS-managed endpoints. "
            "Use Redis notifier for in-process handlers."
        )


# ===================================
# Redis Pub/Sub Implementation
# ===================================

class RedisNotifier(BaseNotifier):
    """Redis Pub/Sub — simple, works with existing Redis."""

    def __init__(self):
        self._redis = None
        self._pubsub = None
        self._handlers: dict[str, list[NotificationHandler]] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        import redis.asyncio as aioredis
        redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        self._pubsub = self._redis.pubsub()
        self._running = True
        logger.info("Redis notifier started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.aclose()
        if self._redis:
            await self._redis.aclose()

    async def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        subject: Optional[str] = None,
    ) -> None:
        if not self._redis:
            raise RuntimeError("Notifier not started")
        message = json.dumps({
            "payload": payload,
            "subject": subject,
            "timestamp": datetime.utcnow().isoformat(),
        }, default=str)
        await self._redis.publish(f"notify:{topic}", message)

    async def subscribe(
        self,
        topic: str,
        handler: NotificationHandler,
    ) -> None:
        self._handlers.setdefault(topic, []).append(handler)
        channel = f"notify:{topic}"
        await self._pubsub.subscribe(channel)
        
        if self._task is None:
            self._task = asyncio.create_task(self._listen_loop())

    async def _listen_loop(self) -> None:
        try:
            while self._running:
                msg = await self._pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if msg and msg["type"] == "message":
                    channel = msg["channel"]
                    topic = channel.replace("notify:", "")
                    data = json.loads(msg["data"])
                    notification = Notification(
                        topic=topic,
                        payload=data["payload"],
                        subject=data.get("subject"),
                    )
                    for handler in self._handlers.get(topic, []):
                        try:
                            await handler(notification)
                        except Exception as e:
                            logger.error(f"Notification handler error: {e}", exc_info=True)
                else:
                    await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass


# ===================================
# In-Memory Implementation (Testing)
# ===================================

class InMemoryNotifier(BaseNotifier):
    """In-memory notifier for tests."""

    def __init__(self):
        self._handlers: dict[str, list[NotificationHandler]] = {}
        self._history: list[Notification] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        self._handlers.clear()
        self._history.clear()

    async def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        subject: Optional[str] = None,
    ) -> None:
        notification = Notification(topic=topic, payload=payload, subject=subject)
        self._history.append(notification)
        for handler in self._handlers.get(topic, []):
            await handler(notification)

    async def subscribe(
        self, topic: str, handler: NotificationHandler
    ) -> None:
        self._handlers.setdefault(topic, []).append(handler)


# ===================================
# Factory
# ===================================

_notifier_instance: Optional[BaseNotifier] = None


def get_notifier() -> BaseNotifier:
    """
    Get the configured notifier instance.
    
    Set NOTIFY_BACKEND env var:
      - "sns"    → SNSNotifier (AWS SNS / LocalStack)
      - "redis"  → RedisNotifier (Redis Pub/Sub)
      - "memory" → InMemoryNotifier (tests only)
    
    Default: "redis"
    """
    global _notifier_instance
    if _notifier_instance is None:
        backend = getattr(settings, "NOTIFY_BACKEND", "redis")
        if backend == "sns":
            _notifier_instance = SNSNotifier()
        elif backend == "redis":
            _notifier_instance = RedisNotifier()
        elif backend == "memory":
            _notifier_instance = InMemoryNotifier()
        else:
            raise ValueError(f"Unknown notify backend: {backend}")
        logger.info(f"Notify backend: {backend}")
    return _notifier_instance
