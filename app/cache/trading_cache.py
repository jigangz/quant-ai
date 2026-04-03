from __future__ import annotations

"""
Trading Session Cache

Persists paper trading portfolio state (positions, cash, trades) to Redis
so state survives API restarts. Falls back to in-memory when Redis is unavailable.
"""

import json
from typing import Any, Dict, List, Optional

from app.cache import get_redis
from app.core.logging import get_logger

logger = get_logger(__name__)

# Redis keys
TRADING_PREFIX = "cache:trading"
PORTFOLIO_KEY = f"{TRADING_PREFIX}:portfolio"
POSITIONS_KEY = f"{TRADING_PREFIX}:positions"
TRADES_KEY = f"{TRADING_PREFIX}:trades"
HISTORY_KEY = f"{TRADING_PREFIX}:history"


class TradingSessionCache:
    """
    Redis-backed persistence for paper trading portfolio state.

    Stores portfolio state so it survives API restarts.
    All operations are best-effort — if Redis is down, the caller
    should continue with its in-memory state.
    """

    async def save_portfolio_state(
        self,
        cash: float,
        starting_cash: float,
        total_realized_pnl: float,
        positions: Dict[str, Dict[str, Any]],
    ) -> bool:
        """
        Save portfolio core state to Redis.

        Args:
            cash: Current cash balance.
            starting_cash: Initial cash amount.
            total_realized_pnl: Total realized P&L.
            positions: Dict of symbol -> {quantity, avg_cost, realized_pnl}.

        Returns:
            True if saved successfully.
        """
        redis = await get_redis()
        if redis is None:
            return False

        try:
            state = {
                "cash": cash,
                "starting_cash": starting_cash,
                "total_realized_pnl": total_realized_pnl,
            }
            pipe = redis.pipeline()
            pipe.set(PORTFOLIO_KEY, json.dumps(state).encode())
            pipe.set(POSITIONS_KEY, json.dumps(positions).encode())
            await pipe.execute()
            logger.debug("Portfolio state saved to Redis")
            return True
        except Exception as e:
            logger.warning(f"Failed to save portfolio state: {e}")
            return False

    async def load_portfolio_state(
        self,
    ) -> Optional[Dict[str, Any]]:
        """
        Load portfolio state from Redis.

        Returns:
            Dict with keys: cash, starting_cash, total_realized_pnl, positions.
            None if no state is cached or Redis is unavailable.
        """
        redis = await get_redis()
        if redis is None:
            return None

        try:
            pipe = redis.pipeline()
            pipe.get(PORTFOLIO_KEY)
            pipe.get(POSITIONS_KEY)
            results = await pipe.execute()

            portfolio_bytes, positions_bytes = results
            if portfolio_bytes is None:
                return None

            state = json.loads(portfolio_bytes)
            state["positions"] = json.loads(positions_bytes) if positions_bytes else {}
            return state
        except Exception as e:
            logger.warning(f"Failed to load portfolio state: {e}")
            return None

    async def save_trades(self, trades: List[Dict[str, Any]]) -> bool:
        """
        Save trade history to Redis.

        Args:
            trades: List of trade dicts (JSON-serializable).

        Returns:
            True if saved successfully.
        """
        redis = await get_redis()
        if redis is None:
            return False

        try:
            await redis.set(TRADES_KEY, json.dumps(trades).encode())
            return True
        except Exception as e:
            logger.warning(f"Failed to save trades: {e}")
            return False

    async def load_trades(self) -> Optional[List[Dict[str, Any]]]:
        """Load trade history from Redis."""
        redis = await get_redis()
        if redis is None:
            return None

        try:
            data = await redis.get(TRADES_KEY)
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            logger.warning(f"Failed to load trades: {e}")
            return None

    async def save_history(self, snapshots: List[Dict[str, Any]]) -> bool:
        """Save portfolio history snapshots to Redis."""
        redis = await get_redis()
        if redis is None:
            return False

        try:
            await redis.set(HISTORY_KEY, json.dumps(snapshots).encode())
            return True
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")
            return False

    async def load_history(self) -> Optional[List[Dict[str, Any]]]:
        """Load portfolio history snapshots from Redis."""
        redis = await get_redis()
        if redis is None:
            return None

        try:
            data = await redis.get(HISTORY_KEY)
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            logger.warning(f"Failed to load history: {e}")
            return None

    async def clear(self) -> bool:
        """Clear all trading session data from Redis."""
        redis = await get_redis()
        if redis is None:
            return False

        try:
            await redis.delete(PORTFOLIO_KEY, POSITIONS_KEY, TRADES_KEY, HISTORY_KEY)
            logger.info("Trading session cache cleared")
            return True
        except Exception as e:
            logger.warning(f"Failed to clear trading cache: {e}")
            return False


# Global singleton
_trading_cache: Optional[TradingSessionCache] = None


def get_trading_cache() -> TradingSessionCache:
    """Get the global trading session cache."""
    global _trading_cache
    if _trading_cache is None:
        _trading_cache = TradingSessionCache()
    return _trading_cache
