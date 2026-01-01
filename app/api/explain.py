from fastapi import APIRouter

from app.services.explain_service import explain

router = APIRouter()


@router.get("/explain")
def explain_api(
    ticker: str,
    lookback: int = 1000,
    top_k: int = 10,
):
    return explain(
        ticker=ticker,
        lookback=lookback,
        top_k=top_k,
    )
