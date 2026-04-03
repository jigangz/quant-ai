"""
Streaming Service Entry Point

Starts all producers and consumers, manages broker lifecycle,
and handles graceful shutdown.

Usage:
    python -m app.streaming.service
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from app.core.settings import settings
from app.infra.broker import get_broker
from app.streaming.news_producer import NewsProducer
from app.streaming.price_producer import PriceProducer
from app.streaming.sentiment_consumer import SentimentConsumer
from app.streaming.signal_consumer import SignalConsumer
from app.streaming.ws_relay import WebSocketRelay

logger = logging.getLogger(__name__)


class StreamingService:
    """
    Orchestrates all streaming producers and consumers.

    Starts the broker, launches producers and consumers as
    concurrent tasks, and shuts everything down on SIGINT/SIGTERM.
    """

    def __init__(self):
        self._broker = get_broker()
        self._components: list = []
        self._running = False

    async def start(self) -> None:
        """Start the broker and all streaming components."""
        logger.info("Starting streaming service...")

        try:
            await self._broker.start()
        except Exception as exc:
            logger.error(f"Broker failed to start: {exc}")
            logger.warning("Streaming service running without broker connection")
            return

        # Create components
        price_producer = PriceProducer(self._broker)
        news_producer = NewsProducer(self._broker)
        sentiment_consumer = SentimentConsumer(self._broker)
        signal_consumer = SignalConsumer(self._broker)
        ws_relay = WebSocketRelay(self._broker)

        self._components = [
            price_producer,
            news_producer,
            sentiment_consumer,
            signal_consumer,
            ws_relay,
        ]

        # Start consumers first (so they're ready before producers publish)
        for component in [sentiment_consumer, signal_consumer, ws_relay]:
            try:
                await component.start()
            except Exception as exc:
                logger.warning(f"Failed to start {component.__class__.__name__}: {exc}")

        # Then start producers
        for component in [price_producer, news_producer]:
            try:
                await component.start()
            except Exception as exc:
                logger.warning(f"Failed to start {component.__class__.__name__}: {exc}")

        self._running = True
        logger.info("Streaming service started (all components running)")

    async def stop(self) -> None:
        """Gracefully stop all components and the broker."""
        if not self._running:
            return
        self._running = False
        logger.info("Stopping streaming service...")

        for component in self._components:
            try:
                await component.stop()
            except Exception as exc:
                logger.warning(f"Error stopping {component.__class__.__name__}: {exc}")

        try:
            await self._broker.stop()
        except Exception as exc:
            logger.warning(f"Error stopping broker: {exc}")

        logger.info("Streaming service stopped")

    async def run_forever(self) -> None:
        """Start the service and block until shutdown signal."""
        await self.start()
        if not self._running:
            return

        stop_event = asyncio.Event()

        def _signal_handler():
            logger.info("Shutdown signal received")
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                pass  # Windows doesn't support add_signal_handler

        await stop_event.wait()
        await self.stop()


async def main() -> None:
    """Entry point for python -m app.streaming.service."""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    service = StreamingService()
    await service.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
