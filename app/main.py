from fastapi import FastAPI
from app.api.health import router as health_router
from app.api.market import router as market_router
from app.api.explain import router as explain_router




app = FastAPI(title="Quant AI Backend v1")

app.include_router(health_router)
app.include_router(market_router)
app.include_router(explain_router)
