"""
Distributed Rate Limiting Middleware

Features:
- Redis-based sliding window counter (supports horizontal scaling)
- Fallback to in-memory rate limiting when Redis is unavailable
- Configurable limits
- Returns 429 when exceeded
- Includes rate limit headers
"""

import asyncio
import time
from collections import defaultdict
from typing import Callable, Optional

import redis.asyncio as aioredis
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.logging import get_logger
from app.core.settings import settings

logger = get_logger(__name__)

# Redis key prefix for rate limit counters
RATE_LIMIT_PREFIX = "ratelimit:"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Distributed rate limiter with Redis sliding window.

    Uses Redis sorted sets for a sliding window counter.
    Falls back to in-memory rate limiting if Redis is unavailable.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_size: int = 10,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.window_size = 60  # seconds

        # In-memory fallback storage: {ip: [timestamps]}
        self._requests: dict[str, list[float]] = defaultdict(list)

        # Async Redis client (lazy initialized)
        self._redis: Optional[aioredis.Redis] = None
        self._redis_available: Optional[bool] = None

    async def _get_redis(self) -> Optional[aioredis.Redis]:
        """Get Redis connection, caching availability status."""
        if self._redis_available is False:
            return None

        if self._redis is None:
            try:
                self._redis = aioredis.from_url(
                    settings.REDIS_URL,
                    decode_responses=False,
                    socket_connect_timeout=1,
                )
                await self._redis.ping()
                self._redis_available = True
                logger.info("Rate limiter using Redis backend")
            except Exception as e:
                logger.warning(f"Rate limiter falling back to in-memory: {e}")
                self._redis_available = False
                self._redis = None
                return None

        return self._redis

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers (behind proxy/load balancer)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct connection
        return request.client.host if request.client else "unknown"

    def _clean_old_requests(self, ip: str, now: float):
        """Remove requests outside the window (in-memory fallback)."""
        cutoff = now - self.window_size
        self._requests[ip] = [
            ts for ts in self._requests[ip] if ts > cutoff
        ]

    def _is_rate_limited_memory(self, ip: str) -> tuple[bool, int, int]:
        """
        In-memory rate limit check (fallback).

        Returns:
            (is_limited, remaining, reset_seconds)
        """
        now = time.time()
        self._clean_old_requests(ip, now)

        request_count = len(self._requests[ip])
        remaining = max(0, self.requests_per_minute - request_count)

        # Calculate reset time
        if self._requests[ip]:
            oldest = min(self._requests[ip])
            reset_seconds = int(self.window_size - (now - oldest))
        else:
            reset_seconds = self.window_size

        if request_count >= self.requests_per_minute:
            return True, remaining, reset_seconds

        return False, remaining, reset_seconds

    async def _is_rate_limited_redis(
        self, ip: str, redis_client: aioredis.Redis
    ) -> tuple[bool, int, int]:
        """
        Redis sliding window rate limit check.

        Uses a sorted set where members are unique request IDs (timestamps)
        and scores are the request timestamps. Old entries are pruned each check.

        Returns:
            (is_limited, remaining, reset_seconds)
        """
        now = time.time()
        key = f"{RATE_LIMIT_PREFIX}{ip}"
        cutoff = now - self.window_size

        pipe = redis_client.pipeline()
        # Remove entries outside the window
        pipe.zremrangebyscore(key, 0, cutoff)
        # Count current entries
        pipe.zcard(key)
        # Add current request (unique member via timestamp)
        pipe.zadd(key, {str(now).encode(): now})
        # Set expiry on the key
        pipe.expire(key, self.window_size + 1)
        results = await pipe.execute()

        request_count = results[1]  # zcard result
        remaining = max(0, self.requests_per_minute - request_count)

        # Get oldest entry for reset calculation
        oldest_entries = await redis_client.zrange(key, 0, 0, withscores=True)
        if oldest_entries:
            oldest_ts = oldest_entries[0][1]
            reset_seconds = max(1, int(self.window_size - (now - oldest_ts)))
        else:
            reset_seconds = self.window_size

        if request_count >= self.requests_per_minute:
            return True, remaining, reset_seconds

        return False, remaining, reset_seconds

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Skip in test environment
        if settings.ENV == "test":
            return await call_next(request)

        ip = self._get_client_ip(request)

        # Try Redis first, fall back to in-memory
        redis_client = await self._get_redis()
        if redis_client is not None:
            try:
                is_limited, remaining, reset_seconds = (
                    await self._is_rate_limited_redis(ip, redis_client)
                )
            except Exception as e:
                logger.warning(f"Redis rate limit error, using in-memory: {e}")
                is_limited, remaining, reset_seconds = (
                    self._is_rate_limited_memory(ip)
                )
                # Record in memory since Redis failed
                self._requests[ip].append(time.time())
        else:
            is_limited, remaining, reset_seconds = self._is_rate_limited_memory(ip)
            # Record this request in memory
            if not is_limited:
                self._requests[ip].append(time.time())

        if is_limited:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too many requests",
                    "message": f"Rate limit exceeded. Try again in {reset_seconds}s",
                    "retry_after": reset_seconds,
                },
                headers={
                    "Retry-After": str(reset_seconds),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_seconds),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining - 1)
        response.headers["X-RateLimit-Reset"] = str(reset_seconds)

        return response
