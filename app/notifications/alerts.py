"""
Alert Notification Service

Publishes market and model alerts via the configured notifier backend.
All publish calls are non-blocking (fire-and-forget via asyncio.create_task).

Topics:
- "alerts" — general alert fan-out (price breakout, indicator, sentiment)
"""

import asyncio
import logging
from typing import Any, Optional

from app.infra.notify import get_notifier

logger = logging.getLogger(__name__)

ALERTS_TOPIC = "alerts"


async def _safe_publish(topic: str, payload: dict[str, Any], subject: Optional[str] = None) -> None:
    """Publish a notification, logging and swallowing errors for graceful degradation."""
    try:
        notifier = get_notifier()
        await notifier.publish(topic, payload, subject=subject)
    except Exception as e:
        logger.warning(f"Failed to publish notification to {topic}: {e}")


def publish_alert_nonblocking(payload: dict[str, Any], subject: Optional[str] = None) -> None:
    """Fire-and-forget alert publish — safe to call from sync or async code."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_safe_publish(ALERTS_TOPIC, payload, subject))
    except RuntimeError:
        # No running loop — skip silently
        logger.debug("No event loop available, skipping alert publish")


# ===================================
# Price Breakout Alerts
# ===================================

async def publish_price_breakout(
    symbol: str,
    current_price: float,
    threshold: float,
    direction: str = "above",
) -> None:
    """Publish a price breakout alert when price crosses a threshold."""
    payload = {
        "type": "price_breakout",
        "symbol": symbol,
        "current_price": current_price,
        "threshold": threshold,
        "direction": direction,
        "message": f"{symbol} broke {direction} ${threshold:.2f} (now ${current_price:.2f})",
    }
    await _safe_publish(ALERTS_TOPIC, payload, subject=f"Price Alert: {symbol}")


# ===================================
# Technical Indicator Alerts
# ===================================

async def publish_rsi_alert(
    symbol: str,
    rsi_value: float,
    condition: str,
) -> None:
    """Publish RSI overbought/oversold alert.

    Args:
        symbol: Ticker symbol
        rsi_value: Current RSI value
        condition: "overbought" (RSI > 70) or "oversold" (RSI < 30)
    """
    payload = {
        "type": "rsi_alert",
        "symbol": symbol,
        "rsi_value": rsi_value,
        "condition": condition,
        "message": f"{symbol} RSI {condition} at {rsi_value:.1f}",
    }
    await _safe_publish(ALERTS_TOPIC, payload, subject=f"RSI Alert: {symbol}")


async def publish_macd_crossover(
    symbol: str,
    macd_value: float,
    signal_value: float,
    crossover_type: str,
) -> None:
    """Publish MACD crossover alert.

    Args:
        symbol: Ticker symbol
        macd_value: Current MACD value
        signal_value: Current signal line value
        crossover_type: "bullish" or "bearish"
    """
    payload = {
        "type": "macd_crossover",
        "symbol": symbol,
        "macd_value": macd_value,
        "signal_value": signal_value,
        "crossover_type": crossover_type,
        "message": f"{symbol} MACD {crossover_type} crossover",
    }
    await _safe_publish(ALERTS_TOPIC, payload, subject=f"MACD Alert: {symbol}")


# ===================================
# Sentiment Anomaly Alerts
# ===================================

async def publish_sentiment_anomaly(
    symbol: str,
    sentiment_score: float,
    previous_score: float,
    shift_magnitude: float,
) -> None:
    """Publish alert for sudden sentiment shift."""
    direction = "positive" if sentiment_score > previous_score else "negative"
    payload = {
        "type": "sentiment_anomaly",
        "symbol": symbol,
        "sentiment_score": sentiment_score,
        "previous_score": previous_score,
        "shift_magnitude": shift_magnitude,
        "direction": direction,
        "message": (
            f"{symbol} sentiment shifted {direction} by {shift_magnitude:.2f} "
            f"({previous_score:.2f} → {sentiment_score:.2f})"
        ),
    }
    await _safe_publish(ALERTS_TOPIC, payload, subject=f"Sentiment Alert: {symbol}")
