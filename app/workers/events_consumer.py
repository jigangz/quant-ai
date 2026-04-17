from __future__ import annotations

"""
Prediction Events Consumer

Subscribes to Kafka 'prediction_events' topic and maintains per-ticker rolling
stats in memory. Exposes /stats/{ticker} endpoint.

Run with: `python -m uvicorn app.workers.events_consumer:app --host 0.0.0.0 --port 8001`

In K8s, this is a separate Deployment using Dockerfile.consumer.
"""

import asyncio
import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Deque, Dict

from fastapi import FastAPI

from app.core.settings import settings
from app.services.prediction_event_publisher import PredictionEvent, TOPIC

logger = logging.getLogger(__name__)

# Per-ticker rolling window of the last 1000 events (in-memory)
_stats: Dict[str, Deque[PredictionEvent]] = defaultdict(lambda: deque(maxlen=1000))
_consumer_state: dict = {"consumer": None, "task": None}


async def _consume_loop() -> None:
    """Background task: consume Kafka messages and update _stats."""
    if settings.BROKER_BACKEND != "kafka":
        logger.info("BROKER_BACKEND != kafka; consumer idle (use direct _stats injection for tests)")
        return

    from aiokafka import AIOKafkaConsumer

    consumer = AIOKafkaConsumer(
        TOPIC,
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id="events-consumer",
        auto_offset_reset="earliest",
    )
    _consumer_state["consumer"] = consumer

    try:
        await consumer.start()
        logger.info("Events consumer subscribed to topic=%s", TOPIC)
        async for msg in consumer:
            try:
                event = PredictionEvent.model_validate_json(msg.value)
                _stats[event.ticker].append(event)
            except Exception as e:
                logger.warning("Malformed Kafka message skipped: %s", e)
    except Exception as e:
        logger.error("Consumer loop error: %s", e)
    finally:
        try:
            await consumer.stop()
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_consume_loop())
    _consumer_state["task"] = task
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    consumer = _consumer_state.get("consumer")
    if consumer is not None:
        try:
            await consumer.stop()
        except Exception:
            pass


app = FastAPI(
    title="Quant AI Events Consumer",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats/{ticker}")
def get_stats(ticker: str):
    ticker = ticker.upper()
    events = list(_stats.get(ticker, []))
    if not events:
        return {"ticker": ticker, "count": 0}
    avg_confidence = sum(e.confidence for e in events) / len(events)
    bullish_ratio = sum(1 for e in events if e.prediction == 1) / len(events)
    return {
        "ticker": ticker,
        "count": len(events),
        "avg_confidence": avg_confidence,
        "bullish_ratio": bullish_ratio,
        "last_prediction_ts": events[-1].timestamp.isoformat(),
    }
