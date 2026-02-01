"""
Simple Rate Limiting Middleware

Features:
- In-memory rate limiting (per-IP)
- Configurable limits
- Returns 429 when exceeded
- Includes rate limit headers

Note: For production with multiple instances, use Redis-based rate limiting.
"""

import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.settings import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter.

    Limits requests per IP address using a sliding window.
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

        # In-memory storage: {ip: [(timestamp, count), ...]}
        self._requests: dict[str, list[float]] = defaultdict(list)

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
        """Remove requests outside the window."""
        cutoff = now - self.window_size
        self._requests[ip] = [
            ts for ts in self._requests[ip] if ts > cutoff
        ]

    def _is_rate_limited(self, ip: str) -> tuple[bool, int, int]:
        """
        Check if IP is rate limited.

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
        is_limited, remaining, reset_seconds = self._is_rate_limited(ip)

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

        # Record this request
        self._requests[ip].append(time.time())

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining - 1)
        response.headers["X-RateLimit-Reset"] = str(reset_seconds)

        return response
