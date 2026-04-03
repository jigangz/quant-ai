"""
Price Relay Consumer (Kafka → WebSocket)

Subscribes to 'market.prices' on the broker and relays each price
update to Redis Pub/Sub, bridging the broker pipeline into the
existing WebSocket infrastructure.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.infra.broker import BaseBroker, BrokerMessage
from app.streaming import TOPIC_MARKET_PRICES
from app.trading.websocket import PRICE_CHANNEL

logger = logging.getLogger(__name__)


class WebSocketRelay:
    """
    Bridges broker price messages to Redis Pub/Sub.

    WebSocket clients already subscribe to the Redis PRICE_CHANNEL
    via PriceStreamer._subscribe_loop. This relay forwards broker
    messages so they appear in the same channel.
    """

    def __init__(self, broker: BaseBroker):
        self._broker = broker
        self._redis = None

    async def start(self) -> None:
        """Subscribe to market.prices and connect to Redis."""
        try:
            from app.cache import get_redis
            self._redis = await get_redis()
        except Exception as exc:
            logger.warning(f"Redis unavailable for WS relay: {exc}")

        await self._broker.subscribe(
            TOPIC_MARKET_PRICES,
            self._handle,
            group_id="quant-ai-ws-relay",
        )
        logger.info("WebSocket relay subscribed to %s", TOPIC_MARKET_PRICES)

    async def _handle(self, message: BrokerMessage) -> None:
        """Relay a price message to Redis Pub/Sub."""
        if self._redis is None:
            return

        data = message.value
        symbol = data.get("symbol")
        if not symbol:
            return

        ws_message: dict[str, Any] = {
            "type": "price_update",
            "data": {
                symbol: {
                    "symbol": symbol,
                    "price": data.get("price"),
                    "volume": data.get("volume", 0),
                    "timestamp": data.get("timestamp"),
                    "change": 0,
                    "change_pct": 0,
                },
            },
        }

        try:
            await self._redis.publish(
                PRICE_CHANNEL, json.dumps(ws_message).encode(),
            )
        except Exception as exc:
            logger.warning(f"Failed to relay price to Redis Pub/Sub: {exc}")
