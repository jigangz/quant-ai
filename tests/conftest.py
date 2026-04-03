"""Root conftest — set environment variables before any app imports."""
from __future__ import annotations

import os

import pytest

# Ensure in-memory backends for all infrastructure during tests
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("CACHE_BACKEND", "memory")
os.environ.setdefault("BROKER_BACKEND", "memory")
os.environ.setdefault("QUEUE_BACKEND", "memory")
os.environ.setdefault("NOTIFY_BACKEND", "memory")
os.environ.setdefault("FUNCTIONS_BACKEND", "local")
os.environ.setdefault("REDIS_URL", "")


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset global singletons between tests to prevent state leakage."""
    yield
    # Reset infra singletons
    try:
        import app.infra.queue as q
        q._queue_instance = None
    except Exception:
        pass
    try:
        import app.infra.broker as b
        b._broker_instance = None
    except Exception:
        pass
    try:
        import app.infra.notify as n
        n._notifier_instance = None
    except Exception:
        pass
    try:
        import app.infra.functions as f
        f._runner_instance = None
    except Exception:
        pass
    try:
        import app.cache as c
        c._redis_client = None
    except Exception:
        pass
