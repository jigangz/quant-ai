from __future__ import annotations

"""
Kafka Real-time Data Pipeline (Phase 2.5 Step 9)

Defines topic constants and exports for the streaming subsystem.
Producers publish market data and news; consumers score sentiment,
generate strategy signals, and relay prices to WebSocket clients.

All components use the broker abstraction from app.infra.broker,
so they work with Kafka, Redis Streams, or InMemoryBroker.
"""

# Topic constants
TOPIC_MARKET_PRICES = "market.prices"
TOPIC_NEWS_RAW = "news.raw"
TOPIC_NEWS_SCORED = "news.scored"
TOPIC_SIGNALS = "signals.generated"

__all__ = [
    "TOPIC_MARKET_PRICES",
    "TOPIC_NEWS_RAW",
    "TOPIC_NEWS_SCORED",
    "TOPIC_SIGNALS",
]
