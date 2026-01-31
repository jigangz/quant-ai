"""
Market Data API

Provides endpoints for fetching OHLCV market data.
"""

from datetime import date
from fastapi import APIRouter, Query, HTTPException

from app.db.prices_repo import get_prices, upsert_prices
from app.providers import get_market_provider

router = APIRouter(prefix="/data", tags=["Market Data"])


@router.get("/market")
def get_market_data(
    ticker: str = Query(..., min_length=1, description="Stock ticker symbol"),
    period: str = Query(
        "1mo", description="Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)"
    ),
    limit: int = Query(30, gt=0, le=500, description="Maximum rows to return"),
):
    """
    Get OHLCV market data for a ticker.

    Data is cached in the database. If not cached, fetches from the market provider.

    Returns:
        List of OHLCV records: [{ticker, date, open, high, low, close, volume}, ...]
    """
    ticker = ticker.upper()

    # Check cache first
    rows = get_prices(ticker, limit)
    if rows:
        return rows

    # Fetch from provider
    provider = get_market_provider()
    df = provider.fetch(ticker, period=period)

    if df.empty:
        raise HTTPException(
            status_code=404, detail=f"No market data found for {ticker}"
        )

    # Cache in database
    upsert_prices(df.to_dict(orient="records"))

    # Return from cache (ensures consistent format)
    return get_prices(ticker, limit)


@router.get("/market/providers")
def list_market_providers():
    """List available market data providers."""
    return {
        "available": ["yahoo"],  # Future: polygon, alpha_vantage
        "current": "yahoo",
    }


@router.get("/sentiment")
def get_sentiment_data(
    ticker: str = Query(..., min_length=1, description="Stock ticker symbol"),
    days: int = Query(30, gt=0, le=365, description="Number of days of history"),
):
    """
    Get sentiment data for a ticker.

    Note: Currently returns mock data. Real sentiment integration planned for V3.

    Returns:
        List of sentiment records
    """
    from app.providers import get_sentiment_provider

    ticker = ticker.upper()
    end_date = date.today()
    start_date = date.today().replace(day=max(1, end_date.day - days))

    provider = get_sentiment_provider()
    df = provider.fetch(ticker, start_date=start_date, end_date=end_date)

    return df.to_dict(orient="records")


@router.get("/news")
def get_news_data(
    ticker: str = Query(..., min_length=1, description="Stock ticker symbol"),
    limit: int = Query(20, gt=0, le=100, description="Maximum articles to return"),
):
    """
    Get news data for a ticker.

    Note: Currently returns mock data. Real news integration planned for V3.

    Returns:
        List of news records
    """
    from app.providers import get_news_provider

    ticker = ticker.upper()

    provider = get_news_provider()
    df = provider.fetch(ticker, limit=limit)

    return df.to_dict(orient="records")
