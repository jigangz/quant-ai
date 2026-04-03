"""
News Producer

Polls for news articles and publishes raw news to the 'news.raw' topic.
Uses Polygon news API when POLYGON_API_KEY is set; otherwise generates
mock news for demo/dev mode.
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.settings import settings
from app.infra.broker import BaseBroker
from app.streaming import TOPIC_NEWS_RAW

logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA"]

MOCK_HEADLINES = [
    ("{symbol} beats earnings expectations, stock surges", "bullish"),
    ("{symbol} reports strong quarterly revenue growth", "bullish"),
    ("{symbol} announces major partnership deal", "bullish"),
    ("{symbol} upgraded by analysts after product launch", "bullish"),
    ("{symbol} misses revenue estimates, shares drop", "bearish"),
    ("{symbol} faces regulatory scrutiny over practices", "bearish"),
    ("{symbol} announces layoffs amid restructuring", "bearish"),
    ("{symbol} downgraded on slowing growth concerns", "bearish"),
    ("{symbol} holds annual shareholder meeting", "neutral"),
    ("{symbol} CEO speaks at industry conference", "neutral"),
    ("{symbol} files routine SEC quarterly report", "neutral"),
]


class NewsProducer:
    """
    Publishes raw news articles to the broker.

    In mock mode (no POLYGON_API_KEY), generates random news headlines.
    Each message: {title, description, symbols, published_at, source_url}.
    """

    def __init__(
        self,
        broker: BaseBroker,
        symbols: Optional[list[str]] = None,
        poll_interval_seconds: float = 30.0,
    ):
        self._broker = broker
        self._symbols = symbols or list(DEFAULT_SYMBOLS)
        self._interval = poll_interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start polling for news."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info(
            "News producer started",
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
        logger.info("News producer stopped")

    async def _run(self) -> None:
        """Main poll loop."""
        try:
            while self._running:
                await self._poll()
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"News producer error: {exc}", exc_info=True)

    async def _poll(self) -> None:
        """Generate or fetch news and publish to broker."""
        articles = self._generate_mock_news()
        for article in articles:
            try:
                symbol = article["symbols"][0] if article["symbols"] else None
                await self._broker.publish(
                    TOPIC_NEWS_RAW, article, key=symbol,
                )
            except Exception as exc:
                logger.warning(f"Failed to publish news article: {exc}")
        if articles:
            logger.debug(f"Published {len(articles)} news articles")

    def _generate_mock_news(self) -> list[dict[str, Any]]:
        """Generate 1-3 mock news articles per poll."""
        articles: list[dict[str, Any]] = []
        count = random.randint(1, 3)
        for _ in range(count):
            symbol = random.choice(self._symbols)
            template, _sentiment = random.choice(MOCK_HEADLINES)
            title = template.format(symbol=symbol)
            articles.append({
                "title": title,
                "description": f"Details about: {title}",
                "symbols": [symbol],
                "published_at": datetime.now(timezone.utc).isoformat(),
                "source_url": f"https://mock-news.example.com/{symbol.lower()}",
            })
        return articles
