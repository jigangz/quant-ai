"""
Strategy Signal Consumer

Subscribes to 'market.prices', runs configured strategies against
recent price data, and publishes generated signals to 'signals.generated'.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.infra.broker import BaseBroker, BrokerMessage
from app.strategies import get_registry
from app.streaming import TOPIC_MARKET_PRICES, TOPIC_SIGNALS

logger = logging.getLogger(__name__)

# Minimum number of price ticks before running strategies
MIN_WINDOW = 30


class SignalConsumer:
    """
    Consumes market prices and produces strategy signals.

    Accumulates price history per symbol, then runs each registered
    strategy. When a strategy produces a non-hold signal, publishes
    to TOPIC_SIGNALS.

    Each signal message: {strategy_name, symbol, signal, confidence, timestamp}.
    """

    def __init__(
        self,
        broker: BaseBroker,
        strategy_names: Optional[list[str]] = None,
        window_size: int = 100,
    ):
        self._broker = broker
        self._strategy_names = strategy_names
        self._window_size = window_size
        # symbol -> list of price dicts
        self._price_history: dict[str, list[dict[str, Any]]] = defaultdict(list)

    async def start(self) -> None:
        """Subscribe to market.prices topic."""
        await self._broker.subscribe(
            TOPIC_MARKET_PRICES,
            self._handle,
            group_id="quant-ai-signals",
        )
        logger.info("Signal consumer subscribed to %s", TOPIC_MARKET_PRICES)

    async def _handle(self, message: BrokerMessage) -> None:
        """Process a price tick and run strategies."""
        data = message.value
        symbol = data.get("symbol")
        price = data.get("price")

        if not symbol or price is None:
            return

        # Accumulate price history
        history = self._price_history[symbol]
        history.append(data)
        if len(history) > self._window_size:
            self._price_history[symbol] = history[-self._window_size:]
            history = self._price_history[symbol]

        # Need enough data to run strategies
        if len(history) < MIN_WINDOW:
            return

        # Build a DataFrame from price history
        df = self._build_dataframe(history)
        if df is None:
            return

        await self._run_strategies(symbol, df)

    def _build_dataframe(self, history: list[dict[str, Any]]) -> Optional[pd.DataFrame]:
        """Build a DataFrame with 'close' column from price history."""
        try:
            prices = [h["price"] for h in history]
            df = pd.DataFrame({"close": prices})
            # Add volume if available
            volumes = [h.get("volume", 0) for h in history]
            df["volume"] = volumes
            # Strategies may need open/high/low — approximate from close
            df["open"] = df["close"].shift(1).fillna(df["close"])
            df["high"] = df[["open", "close"]].max(axis=1)
            df["low"] = df[["open", "close"]].min(axis=1)
            return df
        except Exception as exc:
            logger.warning(f"Failed to build DataFrame: {exc}")
            return None

    async def _run_strategies(self, symbol: str, df: pd.DataFrame) -> None:
        """Run registered strategies and publish signals."""
        registry = get_registry()
        names = self._strategy_names or registry.list_strategies()

        for name in names:
            strategy_cls = registry.get(name)
            if strategy_cls is None:
                continue

            try:
                strategy = strategy_cls()
                missing = strategy.validate_data(df)
                if missing:
                    continue

                signals = strategy.generate_signals(df)
                latest_signal = int(signals.iloc[-1]) if len(signals) > 0 else 0

                if latest_signal == 0:
                    continue

                signal_label = "buy" if latest_signal == 1 else "sell"
                # Confidence heuristic: fraction of recent signals in same direction
                recent = signals.tail(5)
                confidence = float(
                    np.mean(recent == latest_signal)
                ) if len(recent) > 0 else 0.5

                signal_msg: dict[str, Any] = {
                    "strategy_name": name,
                    "symbol": symbol,
                    "signal": signal_label,
                    "confidence": round(confidence, 4),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                await self._broker.publish(
                    TOPIC_SIGNALS, signal_msg, key=symbol,
                )
                logger.info(
                    f"Signal: {name} → {signal_label} {symbol} "
                    f"(confidence={confidence:.2f})"
                )
            except Exception as exc:
                logger.warning(f"Strategy {name} error for {symbol}: {exc}")
