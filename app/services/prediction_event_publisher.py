from __future__ import annotations

"""
Kafka prediction event publisher.

Fire-and-forget publishing to the 'prediction_events' topic. Graceful
degradation: if Kafka is unreachable or BROKER_BACKEND != 'kafka',
publishing is a silent no-op (logs a warning once).

Producer lifecycle is managed by FastAPI lifespan in app/main.py.
"""

import logging
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.core.settings import settings

logger = logging.getLogger(__name__)

TOPIC = "prediction_events"

# Module-level producer state (mutable dict so tests can patch)
_producer_state: dict = {"producer": None, "started": False, "warned_no_kafka": False}


class PredictionEvent(BaseModel):
    """Schema for a prediction event published to Kafka."""

    ticker: str
    prediction: int  # 0 or 1
    confidence: float  # positive-class probability [0, 1]
    model_id: str
    model_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(extra="forbid")


async def start_producer() -> None:
    """Start the global AIOKafkaProducer. No-op if BROKER_BACKEND != kafka."""
    if settings.BROKER_BACKEND != "kafka":
        logger.info("BROKER_BACKEND != kafka; prediction event publisher disabled")
        return

    if _producer_state["started"]:
        return

    try:
        from aiokafka import AIOKafkaProducer

        producer = AIOKafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            acks="all",
        )
        await producer.start()
        _producer_state["producer"] = producer
        _producer_state["started"] = True
        logger.info(
            "Kafka prediction event producer started (bootstrap=%s)",
            settings.KAFKA_BOOTSTRAP_SERVERS,
        )
    except Exception as e:
        logger.warning("Failed to start Kafka producer: %s; will operate as no-op", e)


async def stop_producer() -> None:
    """Stop the global AIOKafkaProducer on shutdown."""
    producer = _producer_state.get("producer")
    if producer is not None:
        try:
            await producer.stop()
        except Exception as e:
            logger.warning("Error stopping Kafka producer: %s", e)
    _producer_state["producer"] = None
    _producer_state["started"] = False


async def publish_prediction_event(event: PredictionEvent) -> None:
    """Fire-and-forget publish of a prediction event.

    If Kafka is unreachable, logs and returns — does NOT raise.
    Keyed by ticker so same ticker goes to same partition for ordering.
    """
    if settings.BROKER_BACKEND != "kafka":
        if not _producer_state["warned_no_kafka"]:
            logger.info("BROKER_BACKEND=%s; prediction events skipped", settings.BROKER_BACKEND)
            _producer_state["warned_no_kafka"] = True
        return

    producer = _producer_state.get("producer")
    if producer is None:
        logger.debug("Kafka producer not ready; skipping event publish")
        return

    try:
        await producer.send_and_wait(
            TOPIC,
            value=event.model_dump_json().encode("utf-8"),
            key=event.ticker.encode("utf-8"),
        )
    except Exception as e:
        logger.warning("Failed to publish prediction event: %s", e)
