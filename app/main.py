from fastapi import FastAPI
from app.api.health import router as health_router

app = FastAPI(title="Quant AI Backend v1")

app.include_router(health_router)
