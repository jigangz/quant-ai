from fastapi import FastAPI

from app.api import (
    health,
    market,
    explain,
    search,
    agents,
    rag,
)


app = FastAPI(title="Quant AI Backend v1")


app.include_router(health.router)
app.include_router(market.router)
app.include_router(explain.router)
app.include_router(search.router)
app.include_router(agents.router)
app.include_router(rag.router)
