from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    health,
    market,
    explain,
    search,
    agents,
    rag,
    predict,   
)

# Create FastAPI application
app = FastAPI(title="Quant AI Backend v1")

# --------------------
# CORS configuration
# Must be added before including routers
# This allows the frontend (e.g. Vite on localhost:5173)
# to make cross-origin requests and pass OPTIONS preflight
# --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],   # Allow GET / POST / OPTIONS, etc.
    allow_headers=["*"],
)

# --------------------
# Register API routers
# --------------------
app.include_router(health.router)
app.include_router(market.router)
app.include_router(explain.router)
app.include_router(search.router)
app.include_router(agents.router)
app.include_router(rag.router)
app.include_router(predict.router)  

