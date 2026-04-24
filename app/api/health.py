from __future__ import annotations

"""Health check endpoint with settings info."""

from fastapi import APIRouter

from app.core.settings import settings

router = APIRouter()


@router.get("/health")
def health():
    """
    Health check endpoint.

    Returns:
        - status: "ok" if service is running
        - settings: public configuration (no secrets)
    """
    return {
        "status": "ok",
        "version": "2.4.0",
        "settings": settings.get_public_settings(),
    }


@router.get("/health/simple")
def health_simple():
    """Simple health check (for load balancers)."""
    return {"status": "ok"}


@router.get("/health/ready")
def health_ready():
    """Readiness probe — returns 200 only if critical deps reachable."""
    try:
        from app.db.engine import get_engine
        from sqlalchemy import text

        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="database unreachable")
    return {"status": "ready"}
