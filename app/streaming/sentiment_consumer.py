"""
Sentiment Scoring Consumer

Subscribes to 'news.raw', runs sentiment analysis on each article,
and publishes scored results to 'news.scored'.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.functions.sentiment_analyzer import handle_sentiment_analysis
from app.infra.broker import BaseBroker, BrokerMessage
from app.streaming import TOPIC_NEWS_RAW, TOPIC_NEWS_SCORED

logger = logging.getLogger(__name__)


class SentimentConsumer:
    """
    Consumes raw news and produces sentiment-scored news.

    Subscribes to TOPIC_NEWS_RAW, scores each article via
    handle_sentiment_analysis, and publishes to TOPIC_NEWS_SCORED.

    Each output message contains the original news fields plus:
    {sentiment_score, sentiment_label, analyzed_at}.
    """

    def __init__(self, broker: BaseBroker):
        self._broker = broker

    async def start(self) -> None:
        """Subscribe to news.raw topic."""
        await self._broker.subscribe(
            TOPIC_NEWS_RAW,
            self._handle,
            group_id="quant-ai-sentiment",
        )
        logger.info("Sentiment consumer subscribed to %s", TOPIC_NEWS_RAW)

    async def _handle(self, message: BrokerMessage) -> None:
        """Score a single news article and publish the result."""
        news = message.value
        text = news.get("title", "") + " " + news.get("description", "")
        text = text.strip()

        if not text:
            logger.debug("Skipping empty news article")
            return

        try:
            result = await handle_sentiment_analysis({"text": text})
        except Exception as exc:
            logger.error(f"Sentiment analysis failed: {exc}", exc_info=True)
            return

        score = result.get("score", 0.0)
        label = _score_to_label(score)

        scored_message: dict[str, Any] = {
            **news,
            "sentiment_score": score,
            "sentiment_label": label,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            symbol = news.get("symbols", [None])[0] if news.get("symbols") else None
            await self._broker.publish(
                TOPIC_NEWS_SCORED, scored_message, key=symbol,
            )
        except Exception as exc:
            logger.warning(f"Failed to publish scored news: {exc}")


def _score_to_label(score: float) -> str:
    """Convert numeric score to human-readable label."""
    if score >= 0.3:
        return "bullish"
    if score <= -0.3:
        return "bearish"
    return "neutral"
