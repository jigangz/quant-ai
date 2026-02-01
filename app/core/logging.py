"""
Structured Logging with Request ID Propagation

Features:
- JSON structured logs in production
- Request ID in all log entries
- Timing information
- Context propagation
"""

import json
import logging
import sys
import time
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from app.core.settings import settings

# Context variable for request_id propagation
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter with request_id."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get("-"),
        }

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add source location in debug mode
        if settings.DEBUG:
            log_data["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable formatter with request_id."""

    def format(self, record: logging.LogRecord) -> str:
        request_id = request_id_ctx.get("-")
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"{timestamp} [{record.levelname}] "
            f"[{request_id}] {record.name}: {record.getMessage()}"
        )


def setup_logging():
    """Configure structured logging based on settings."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, settings.LOG_LEVEL))

    # Choose formatter based on LOG_FORMAT setting
    if settings.LOG_FORMAT == "json":
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(TextFormatter())

    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding extra data to logs."""

    def __init__(self, **kwargs: Any):
        self.extra_data = kwargs
        self.token = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **extra: Any,
):
    """Log with extra context data."""
    record = logger.makeRecord(
        logger.name,
        level,
        "",
        0,
        message,
        (),
        None,
    )
    record.extra_data = extra
    logger.handle(record)


# Timer utility for performance logging
class Timer:
    """Simple timer for measuring operation duration."""

    def __init__(self, name: str, logger: logging.Logger | None = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration_ms = (self.end_time - self.start_time) * 1000
        self.logger.info(
            f"{self.name} completed",
            extra={"extra_data": {"duration_ms": round(duration_ms, 2)}},
        )

    @property
    def elapsed_ms(self) -> float:
        if self.start_time is None:
            return 0
        end = self.end_time or time.perf_counter()
        return (end - self.start_time) * 1000
