"""
Market Price Producer

Publishes real-time price ticks to the broker on the 'market.prices' topic.
Uses Polygon WebSocket when POLYGON_API_KEY is set; otherwise generates
mock prices via PriceGenerator for demo/dev mode.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.settings import settings
from app.infra.broker import BaseBroker
from app.streaming import TOPIC_MARKET_PRICES
from app.trading.websocket import PriceGenerator

logger = logging.getLogger(__name__)

# Default symbols for mock mode
DEFAULT_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA", "SPY"]


class PriceProducer:
    """
    Publishes price updates to the broker.

    In mock mode (no POLYGON_API_KEY), generates prices using PriceGenerator.
    Each message: {symbol, price, volume, timestamp, source}.
    """

    def __init__(
        self,
        broker: BaseBroker,
        symbols: Optional[list[str]] = None,
        interval_seconds: float = 1.0,
    ):
        self._broker = broker
        self._symbols = symbols or list(DEFAULT_SYMBOLS)
        self._interval = interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._generator = PriceGenerator()

    async def start(self) -> None:
        """Start producing price messages."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info(
            "Price producer started",
            extra={"symbols": self._symbols, "interval": self._interval},
        )

    async def stop(self) -> None:
        """Stop the producer gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Price producer stopped")

    async def _run(self) -> None:
        """Main loop: generate ticks and publish."""
        try:
            while self._running:
                await self._tick()
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"Price producer error: {exc}", exc_info=True)

    async def _tick(self) -> None:
        """Generate one round of price updates and publish."""
        updates = self._generator.tick(self._symbols)
        for symbol, update in updates.items():
            message: dict[str, Any] = {
                "symbol": symbol,
                "price": update.price,
                "volume": update.volume,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "mock",
            }
            try:
                await self._broker.publish(
                    TOPIC_MARKET_PRICES, message, key=symbol,
                )
            except Exception as exc:
                logger.warning(f"Failed to publish price for {symbol}: {exc}")

    async def publish_once(self, symbol: str, price: float, volume: int = 0) -> None:
        """Publish a single price message (useful for external callers)."""
        message: dict[str, Any] = {
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "external",
        }
        await self._broker.publish(TOPIC_MARKET_PRICES, message, key=symbol)
