"""
Quant AI Backend - FastAPI Application
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.settings import settings
from app.core.logging import setup_logging, get_logger, request_id_ctx
from app.middleware import RateLimitMiddleware, RequestContextMiddleware
from app.api import (
    health,
    market,
    explain,
    search,
    agents,
    rag,
    predict,
    features,
    train,
    models,
    backtest,
    runs,
)

# ===================================
# Setup Structured Logging
# ===================================
setup_logging()
logger = get_logger(__name__)


# ===================================
# Lifespan Events
# ===================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info(
        "Starting Quant AI Backend",
        extra={
            "extra_data": {
                "env": settings.ENV,
                "providers": settings.providers_list,
                "storage": settings.STORAGE_BACKEND,
            }
        },
    )
    yield
    # Shutdown
    logger.info("Shutting down Quant AI Backend")


# ===================================
# Create FastAPI Application
# ===================================
app = FastAPI(
    title="Quant AI Backend",
    version="2.0.0",
    description="Quantitative research and prediction platform",
    lifespan=lifespan,
)


# ===================================
# Middleware (order matters: last added = first executed)
# ===================================

# CORS (outermost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=settings.RATE_LIMIT_PER_MINUTE,
)

# Request Context (innermost - sets up logging context)
app.add_middleware(RequestContextMiddleware)


# ===================================
# Exception Handler
# ===================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with request_id."""
    request_id = getattr(request.state, "request_id", request_id_ctx.get("-"))

    logger.error(
        f"Unhandled error: {exc}",
        extra={
            "extra_data": {
                "error_type": type(exc).__name__,
                "path": request.url.path,
            }
        },
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
            "detail": str(exc) if settings.DEBUG else None,
        },
        headers={"X-Request-ID": request_id},
    )


# ===================================
# Register API Routers
# ===================================
app.include_router(health.router, tags=["Health"])
app.include_router(market.router, tags=["Market Data"])
app.include_router(features.router, tags=["Features"])
app.include_router(train.router, tags=["Training"])
app.include_router(runs.router, tags=["Training Runs"])
app.include_router(models.router, tags=["Model Registry"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(backtest.router, tags=["Backtest"])
app.include_router(explain.router, tags=["Explainability"])
app.include_router(search.router, tags=["Search"])
app.include_router(agents.router, tags=["Agents"])
app.include_router(rag.router, tags=["RAG"])


# ===================================
# Root Endpoint
# ===================================
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Quant AI Backend",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }
