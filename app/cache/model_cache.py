from __future__ import annotations

"""
Model Artifact Cache

Caches loaded ML model objects in Redis (pickle serialized, with TTL)
to avoid re-loading models from disk/S3 on every predict request.
Falls back to in-memory LRU cache when Redis is unavailable.
"""

import hashlib
import pickle
from typing import Any, Dict, Optional

from app.cache import get_redis
from app.core.logging import get_logger

logger = get_logger(__name__)

# Redis key prefix and default TTL
MODEL_CACHE_PREFIX = "cache:model:"
DEFAULT_TTL_SECONDS = 3600  # 1 hour


class ModelArtifactCache:
    """
    Cache for loaded ML model objects.

    Serializes models with pickle and stores in Redis with TTL.
    Falls back to a simple in-memory dict when Redis is unavailable.
    """

    def __init__(self, ttl: int = DEFAULT_TTL_SECONDS, max_mem_entries: int = 10):
        self._ttl = ttl
        self._max_mem_entries = max_mem_entries
        # In-memory fallback: key -> model object
        self._mem_cache: Dict[str, Any] = {}

    def _cache_key(self, model_path: str) -> str:
        """Generate a Redis key from model path."""
        path_hash = hashlib.sha256(model_path.encode()).hexdigest()[:16]
        return f"{MODEL_CACHE_PREFIX}{path_hash}"

    async def get(self, model_path: str) -> Optional[Any]:
        """
        Get a cached model object.

        Args:
            model_path: Path or identifier for the model.

        Returns:
            The model object if cached, None otherwise.
        """
        redis = await get_redis()

        if redis is not None:
            try:
                data = await redis.get(self._cache_key(model_path))
                if data is not None:
                    model = pickle.loads(data)
                    logger.debug(f"Model cache hit: {model_path}")
                    return model
            except Exception as e:
                logger.warning(f"Redis read error in model cache: {e}")

        # In-memory fallback
        model = self._mem_cache.get(model_path)
        if model is not None:
            logger.debug(f"Model cache hit (memory): {model_path}")
        return model

    async def set(self, model_path: str, model: Any) -> None:
        """
        Cache a model object.

        Args:
            model_path: Path or identifier for the model.
            model: The model object to cache (must be picklable).
        """
        redis = await get_redis()

        if redis is not None:
            try:
                data = pickle.dumps(model)
                await redis.set(self._cache_key(model_path), data, ex=self._ttl)
                logger.debug(f"Model cached in Redis: {model_path}")
                return
            except Exception as e:
                logger.warning(f"Redis write error in model cache: {e}")

        # In-memory fallback with simple eviction
        if len(self._mem_cache) >= self._max_mem_entries:
            # Remove oldest entry
            oldest_key = next(iter(self._mem_cache))
            del self._mem_cache[oldest_key]
        self._mem_cache[model_path] = model
        logger.debug(f"Model cached in memory: {model_path}")

    async def invalidate(self, model_path: str) -> None:
        """Remove a model from cache."""
        redis = await get_redis()

        if redis is not None:
            try:
                await redis.delete(self._cache_key(model_path))
            except Exception as e:
                logger.warning(f"Redis delete error in model cache: {e}")

        self._mem_cache.pop(model_path, None)

    async def clear(self) -> None:
        """Clear all cached models."""
        self._mem_cache.clear()

        redis = await get_redis()
        if redis is not None:
            try:
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=f"{MODEL_CACHE_PREFIX}*", count=100
                    )
                    if keys:
                        await redis.delete(*keys)
                    if cursor == 0:
                        break
                logger.info("Model cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear model cache: {e}")


# Global singleton
_model_cache: Optional[ModelArtifactCache] = None


def get_model_cache() -> ModelArtifactCache:
    """Get the global model artifact cache."""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelArtifactCache()
    return _model_cache
