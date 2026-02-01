"""
Request Context Middleware

Propagates request_id and timing information throughout the request lifecycle.
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger, request_id_ctx

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request context propagation.

    Features:
    - Generates or extracts request_id
    - Sets context variable for logging
    - Logs request start/end with timing
    - Adds headers to response
    """

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Extract or generate request_id
        request_id = request.headers.get(
            "X-Request-ID",
            str(uuid.uuid4())[:8]
        )

        # Set context variable for structured logging
        token = request_id_ctx.set(request_id)

        # Store in request state for handlers
        request.state.request_id = request_id

        # Record start time
        start_time = time.perf_counter()

        # Log request start
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "extra_data": {
                    "method": request.method,
                    "path": request.url.path,
                    "query": str(request.query_params),
                    "client_ip": self._get_client_ip(request),
                }
            },
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log request completion
            logger.info(
                f"Request completed: {response.status_code}",
                extra={
                    "extra_data": {
                        "status_code": response.status_code,
                        "duration_ms": round(duration_ms, 2),
                    }
                },
            )

            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

            return response

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"Request failed: {e}",
                extra={
                    "extra_data": {
                        "error": str(e),
                        "duration_ms": round(duration_ms, 2),
                    }
                },
                exc_info=True,
            )
            raise

        finally:
            # Reset context
            request_id_ctx.reset(token)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"
