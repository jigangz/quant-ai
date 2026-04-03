"""
Market Price Cache

Caches hot stock prices in Redis Hash with short TTL (5 seconds).
Falls back to direct provider fetch when Redis is unavailable.
"""

import json
import time
from typing import Dict, Optional

from app.cache import get_redis
from app.core.logging import get_logger

logger = get_logger(__name__)

# Redis key for the price hash
PRICE_HASH_KEY = "cache:market:prices"
PRICE_TS_HASH_KEY = "cache:market:prices:ts"
PRICE_TTL_SECONDS = 5


class MarketPriceCache:
    """
    Caching wrapper for market price lookups.

    Stores prices in a Redis Hash with per-entry TTL tracking.
    When a cached price is fresh (< TTL), returns it immediately.
    Otherwise fetches from the underlying provider and caches the result.
    """

    def __init__(self, ttl: int = PRICE_TTL_SECONDS):
        self._ttl = ttl
        # In-memory fallback cache
        self._mem_cache: Dict[str, tuple[float, float]] = {}  # symbol -> (price, timestamp)

    async def get_price(self, symbol: str) -> Optional[float]:
        """
        Get a cached price for a symbol.

        Returns:
            Cached price if fresh, None if not cached or expired.
        """
        symbol = symbol.upper()
        redis = await get_redis()

        if redis is not None:
            try:
                price_bytes = await redis.hget(PRICE_HASH_KEY, symbol)
                ts_bytes = await redis.hget(PRICE_TS_HASH_KEY, symbol)
                if price_bytes is not None and ts_bytes is not None:
                    ts = float(ts_bytes)
                    if time.time() - ts < self._ttl:
                        return float(price_bytes)
                return None
            except Exception as e:
                logger.warning(f"Redis read error in market cache: {e}")

        # In-memory fallback
        entry = self._mem_cache.get(symbol)
        if entry and time.time() - entry[1] < self._ttl:
            return entry[0]
        return None

    async def set_price(self, symbol: str, price: float) -> None:
        """Cache a price for a symbol."""
        symbol = symbol.upper()
        now = time.time()
        redis = await get_redis()

        if redis is not None:
            try:
                pipe = redis.pipeline()
                pipe.hset(PRICE_HASH_KEY, symbol, str(price))
                pipe.hset(PRICE_TS_HASH_KEY, symbol, str(now))
                # Set TTL on the hash keys as a safety net
                pipe.expire(PRICE_HASH_KEY, self._ttl * 10)
                pipe.expire(PRICE_TS_HASH_KEY, self._ttl * 10)
                await pipe.execute()
                return
            except Exception as e:
                logger.warning(f"Redis write error in market cache: {e}")

        # In-memory fallback
        self._mem_cache[symbol] = (price, now)

    async def set_prices(self, prices: Dict[str, float]) -> None:
        """Cache multiple prices at once."""
        now = time.time()
        redis = await get_redis()

        if redis is not None:
            try:
                pipe = redis.pipeline()
                price_mapping = {k.upper(): str(v) for k, v in prices.items()}
                ts_mapping = {k.upper(): str(now) for k in prices}
                pipe.hset(PRICE_HASH_KEY, mapping=price_mapping)
                pipe.hset(PRICE_TS_HASH_KEY, mapping=ts_mapping)
                pipe.expire(PRICE_HASH_KEY, self._ttl * 10)
                pipe.expire(PRICE_TS_HASH_KEY, self._ttl * 10)
                await pipe.execute()
                return
            except Exception as e:
                logger.warning(f"Redis write error in market cache: {e}")

        # In-memory fallback
        for symbol, price in prices.items():
            self._mem_cache[symbol.upper()] = (price, now)

    async def invalidate(self, symbol: str) -> None:
        """Remove a symbol from cache."""
        symbol = symbol.upper()
        redis = await get_redis()

        if redis is not None:
            try:
                pipe = redis.pipeline()
                pipe.hdel(PRICE_HASH_KEY, symbol)
                pipe.hdel(PRICE_TS_HASH_KEY, symbol)
                await pipe.execute()
                return
            except Exception as e:
                logger.warning(f"Redis delete error in market cache: {e}")

        self._mem_cache.pop(symbol, None)


# Global singleton
_market_cache: Optional[MarketPriceCache] = None


def get_market_cache() -> MarketPriceCache:
    """Get the global market price cache."""
    global _market_cache
    if _market_cache is None:
        _market_cache = MarketPriceCache()
    return _market_cache
