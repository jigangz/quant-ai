from __future__ import annotations

# app/db/engine.py
# Lazy-loaded engine — connection is only made on first actual DB call,
# not at import time. Prevents crash when Supabase is unreachable.

from sqlalchemy import create_engine
from app.core.config import settings

_engine = None


def get_engine():
    """Return the SQLAlchemy engine, creating it lazily on first call."""
    global _engine
    if _engine is None:
        # Supabase's transaction pooler (port 6543) does NOT support prepared
        # statements — each transaction may land on a different backend, so
        # psycopg3's auto-prepare (after 5 reuses of a statement) throws
        # "prepared statement already exists" on bulk operations. Disable it.
        _engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            connect_args={"prepare_threshold": None},
        )
    return _engine


# Backward compat — existing code uses `from app.db.engine import engine`
# This module-level reference triggers lazy init on first attribute access.
class _LazyEngine:
    """Proxy that defers engine creation until first use."""

    def __getattr__(self, name):
        return getattr(get_engine(), name)


engine = _LazyEngine()
