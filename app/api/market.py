from fastapi import APIRouter, Query, HTTPException
from typing import List

from app.db.prices_repo import get_prices, upsert_prices
from app.providers.yahoo import fetch_ohlcv

router = APIRouter(prefix="/data", tags=["market"])


@router.get("/market")
def get_market_data(
    ticker: str = Query(..., min_length=1),
    period: str = Query("1mo"),
    limit: int = Query(30, gt=0, le=500),
):
    ticker = ticker.upper()

    
    rows = get_prices(ticker, limit)

    if rows:
        return rows

    
    df = fetch_ohlcv(ticker, period)

    if df.empty:
        raise HTTPException(status_code=404, detail="No market data found")

    upsert_prices(df.to_dict(orient="records"))

   
    return get_prices(ticker, limit)
