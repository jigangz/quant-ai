"""Root conftest — set environment variables before any app imports."""
from __future__ import annotations

import os

import pytest

# Ensure in-memory backends for all infrastructure during tests
# ALL env vars must be set BEFORE any app imports
os.environ.setdefault("ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_quant.db")
os.environ.setdefault("STORAGE_BACKEND", "local")
os.environ.setdefault("STORAGE_LOCAL_PATH", "./test_artifacts")
os.environ.setdefault("CACHE_BACKEND", "memory")
os.environ.setdefault("BROKER_BACKEND", "memory")
os.environ.setdefault("QUEUE_BACKEND", "memory")
os.environ.setdefault("NOTIFY_BACKEND", "memory")
os.environ.setdefault("FUNCTIONS_BACKEND", "local")
os.environ.setdefault("REDIS_URL", "")

# Pre-import modules that test_functions.py would mock via setdefault,
# ensuring real implementations win over MagicMock replacements.
import app.db.model_registry  # noqa: E402
import app.cache  # noqa: E402
import app.cache.market_cache  # noqa: E402
import app.db.prices_repo  # noqa: E402
import app.services.model_cache  # noqa: E402
import app.middleware  # noqa: E402
import app.providers.base  # noqa: E402
import app.explain.shap_explainer  # noqa: E402
import app.explain.shap_to_text  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _create_db_tables():
    """Ensure SQLite tables exist for all tests."""
    from sqlalchemy import text
    from app.db.engine import engine
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL, high REAL, low REAL, close REAL,
                volume INTEGER,
                UNIQUE(ticker, date)
            )
        """))
    yield


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
