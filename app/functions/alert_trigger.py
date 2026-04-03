"""
Alert Trigger Function

Evaluates price data against configured alert rules and publishes
triggered alerts via the notifier.

Supported rules: price_breakout, rsi_threshold, macd_crossover.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.infra.notify import get_notifier
from app.notifications.alerts import (
    publish_macd_crossover,
    publish_price_breakout,
    publish_rsi_alert,
)

logger = logging.getLogger(__name__)


def _evaluate_price_breakout(
    symbol: str,
    price: float,
    rule: dict[str, Any],
) -> dict[str, Any] | None:
    """Check if price crossed a threshold."""
    threshold = rule.get("threshold")
    direction = rule.get("direction", "above")
    if threshold is None:
        return None

    if direction == "above" and price > threshold:
        return {
            "type": "price_breakout",
            "symbol": symbol,
            "current_price": price,
            "threshold": threshold,
            "direction": "above",
        }
    elif direction == "below" and price < threshold:
        return {
            "type": "price_breakout",
            "symbol": symbol,
            "current_price": price,
            "threshold": threshold,
            "direction": "below",
        }
    return None


def _evaluate_rsi_threshold(
    symbol: str,
    rsi_value: float | None,
    rule: dict[str, Any],
) -> dict[str, Any] | None:
    """Check if RSI is overbought or oversold."""
    if rsi_value is None:
        return None

    overbought = rule.get("overbought", 70)
    oversold = rule.get("oversold", 30)

    if rsi_value > overbought:
        return {
            "type": "rsi_alert",
            "symbol": symbol,
            "rsi_value": rsi_value,
            "condition": "overbought",
        }
    elif rsi_value < oversold:
        return {
            "type": "rsi_alert",
            "symbol": symbol,
            "rsi_value": rsi_value,
            "condition": "oversold",
        }
    return None


def _evaluate_macd_crossover(
    symbol: str,
    macd_value: float | None,
    signal_value: float | None,
    prev_macd: float | None,
    prev_signal: float | None,
    rule: dict[str, Any],
) -> dict[str, Any] | None:
    """Check for MACD/signal line crossover."""
    if any(v is None for v in (macd_value, signal_value, prev_macd, prev_signal)):
        return None

    was_below = prev_macd < prev_signal
    is_above = macd_value > signal_value

    if was_below and is_above:
        return {
            "type": "macd_crossover",
            "symbol": symbol,
            "macd_value": macd_value,
            "signal_value": signal_value,
            "crossover_type": "bullish",
        }
    elif not was_below and not is_above:
        return {
            "type": "macd_crossover",
            "symbol": symbol,
            "macd_value": macd_value,
            "signal_value": signal_value,
            "crossover_type": "bearish",
        }
    return None


# ---------------------------------------------------------------------------
# Function handler
# ---------------------------------------------------------------------------

async def handle_alert_trigger(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Alert trigger function handler.

    Payload:
        symbol (str): Ticker symbol.
        price (float): Current price.
        rsi (float|None): Current RSI value.
        macd (float|None): Current MACD value.
        signal (float|None): Current signal line value.
        prev_macd (float|None): Previous MACD value.
        prev_signal (float|None): Previous signal line value.
        rules (list[dict]): Alert rules to evaluate.
            Each rule: {"type": "price_breakout"|"rsi_threshold"|"macd_crossover", ...}

    Returns:
        triggered (list[dict]): List of triggered alerts.
    """
    symbol = payload.get("symbol", "")
    if not symbol:
        return {"success": False, "error": "Missing 'symbol' in payload"}

    price = payload.get("price", 0.0)
    rsi = payload.get("rsi")
    macd = payload.get("macd")
    signal = payload.get("signal")
    prev_macd = payload.get("prev_macd")
    prev_signal = payload.get("prev_signal")
    rules = payload.get("rules", [])

    triggered: list[dict[str, Any]] = []

    for rule in rules:
        rule_type = rule.get("type", "")
        result = None

        if rule_type == "price_breakout":
            result = _evaluate_price_breakout(symbol, price, rule)
        elif rule_type == "rsi_threshold":
            result = _evaluate_rsi_threshold(symbol, rsi, rule)
        elif rule_type == "macd_crossover":
            result = _evaluate_macd_crossover(
                symbol, macd, signal, prev_macd, prev_signal, rule,
            )
        else:
            logger.warning(f"Unknown alert rule type: {rule_type}")

        if result is not None:
            triggered.append(result)

    # Publish triggered alerts via notifier
    for alert in triggered:
        try:
            alert_type = alert["type"]
            if alert_type == "price_breakout":
                await publish_price_breakout(
                    alert["symbol"],
                    alert["current_price"],
                    alert["threshold"],
                    alert["direction"],
                )
            elif alert_type == "rsi_alert":
                await publish_rsi_alert(
                    alert["symbol"],
                    alert["rsi_value"],
                    alert["condition"],
                )
            elif alert_type == "macd_crossover":
                await publish_macd_crossover(
                    alert["symbol"],
                    alert["macd_value"],
                    alert["signal_value"],
                    alert["crossover_type"],
                )
        except Exception as e:
            logger.warning(f"Failed to publish alert: {e}")

    return {"success": True, "triggered": triggered, "evaluated": len(rules)}
