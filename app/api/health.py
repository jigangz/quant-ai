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
        "settings": settings.get_public_settings(),
    }


@router.get("/health/simple")
def health_simple():
    """Simple health check (for load balancers)."""
    return {"status": "ok"}
