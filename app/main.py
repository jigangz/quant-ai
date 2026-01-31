"""
Quant AI Backend - FastAPI Application
"""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.settings import settings
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
)

# ===================================
# Logging Configuration
# ===================================
LOG_FORMAT_TEXT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_FORMAT_JSON = (
    '{"time":"%(asctime)s","level":"%(levelname)s",'
    '"logger":"%(name)s","message":"%(message)s"}'
)
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=LOG_FORMAT_TEXT if settings.LOG_FORMAT == "text" else LOG_FORMAT_JSON,
)
logger = logging.getLogger(__name__)


# ===================================
# Lifespan Events
# ===================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info(f"Starting Quant AI Backend (env={settings.ENV})")
    logger.info(f"Providers enabled: {settings.providers_list}")
    logger.info(f"Storage backend: {settings.STORAGE_BACKEND}")
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
# Middleware
# ===================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request for tracing."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"[{request_id}] Unhandled error: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
            "detail": str(exc) if settings.DEBUG else None,
        },
    )


# ===================================
# Register API Routers
# ===================================
app.include_router(health.router, tags=["Health"])
app.include_router(market.router, tags=["Market Data"])
app.include_router(features.router, tags=["Features"])
app.include_router(train.router, tags=["Training"])
app.include_router(models.router, tags=["Model Registry"])
app.include_router(predict.router, tags=["Prediction"])
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
