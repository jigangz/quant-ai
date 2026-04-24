from __future__ import annotations

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
    news,
    strategies,
    trading,
    functions,
    optimize,
    signal,
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
    # Register serverless functions
    from app.functions import register_all_functions
    register_all_functions()
    # Start Kafka prediction event producer (no-op if BROKER_BACKEND != kafka)
    from app.services.prediction_event_publisher import start_producer, stop_producer
    await start_producer()
    yield
    # Shutdown
    await stop_producer()
    logger.info("Shutting down Quant AI Backend")


# ===================================
# Create FastAPI Application
# ===================================
app = FastAPI(
    title="Quant AI Backend",
    version="2.4.0",
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
# Prometheus Instrumentation
# ===================================
from prometheus_fastapi_instrumentator import Instrumentator

# Import custom metrics so they register with the default registry
from app.core import metrics as _metrics  # noqa: F401

Instrumentator(
    should_group_status_codes=True,
    excluded_handlers=["/metrics"],
).instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")


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
app.include_router(strategies.router, tags=["Strategies"])
app.include_router(explain.router, tags=["Explainability"])
app.include_router(search.router, tags=["Search"])
app.include_router(agents.router, tags=["Agents"])
app.include_router(rag.router, tags=["RAG"])
app.include_router(news.router, tags=["News Data"])
app.include_router(trading.router, tags=["Paper Trading"])
app.include_router(functions.router, tags=["Functions"])
app.include_router(optimize.router)
app.include_router(signal.router, tags=["Meta-Labeling"])


# ===================================
# Root Endpoint
# ===================================
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Quant AI Backend",
        "version": "2.4.0",
        "docs": "/docs",
        "health": "/health",
    }
