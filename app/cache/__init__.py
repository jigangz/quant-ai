from __future__ import annotations

"""
Redis Cache Layer

Shared async Redis connection and utilities for all cache modules.
Gracefully degrades when Redis is unavailable.
"""

import redis.asyncio as aioredis
from typing import Optional

from app.core.logging import get_logger
from app.core.settings import settings

logger = get_logger(__name__)

# Global async Redis connection pool
_redis_client: Optional[aioredis.Redis] = None


async def get_redis() -> Optional[aioredis.Redis]:
    """
    Get the shared async Redis client.

    Returns None if Redis is unavailable, allowing callers to fall back
    to in-memory behavior.
    """
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = aioredis.from_url(
                settings.REDIS_URL,
                decode_responses=False,
                socket_connect_timeout=2,
            )
            await _redis_client.ping()
            logger.info("Redis cache connection established")
        except Exception as e:
            logger.warning(f"Redis unavailable, falling back to in-memory: {e}")
            _redis_client = None
            return None
    try:
        await _redis_client.ping()
    except Exception:
        logger.warning("Redis connection lost, resetting")
        _redis_client = None
        return None
    return _redis_client


async def close_redis() -> None:
    """Close the shared Redis connection."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis cache connection closed")
