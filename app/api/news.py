"""
News Data API

Provides endpoints for news articles, sentiment analysis,
and related features.
"""

from datetime import date

from fastapi import APIRouter, Query, HTTPException

from app.services.news_service import (
    fetch_and_store_news,
    get_news_for_ticker,
    get_news_sentiment_summary,
    get_similar_days,
)
from app.db.news_repo import get_news_categories

router = APIRouter(prefix="/data", tags=["News Data"])


@router.get("/news/{ticker}")
def get_news(
    ticker: str,
    date: str | None = Query(None, description="Filter by date (YYYY-MM-DD)"),
    limit: int = Query(50, gt=0, le=500, description="Maximum articles"),
    category: str | None = Query(None, description="Filter by category"),
):
    """
    Get news articles for a ticker.

    Returns cached data if available; otherwise fetches from the configured
    news provider.

    Returns:
        List of news records.
    """
    ticker = ticker.upper()
    rows = get_news_for_ticker(ticker, date=date, limit=limit, category=category)
    return rows


@router.get("/news/{ticker}/categories")
def get_categories(ticker: str):
    """
    Get news categories and their counts for a ticker.

    Returns:
        List of {category, count} dicts.
    """
    ticker = ticker.upper()
    return get_news_categories(ticker)


@router.get("/news/{ticker}/sentiment-summary")
def sentiment_summary(
    ticker: str,
    date: str = Query(..., description="Date (YYYY-MM-DD)"),
):
    """
    Get sentiment summary for a ticker on a specific date.

    Returns:
        Sentiment summary with avg score, counts, and category breakdown.
    """
    ticker = ticker.upper()
    return get_news_sentiment_summary(ticker, date)


@router.get("/news/{ticker}/similar-days")
def similar_days(
    ticker: str,
    date: str = Query(..., description="Reference date (YYYY-MM-DD)"),
    top_k: int = Query(5, gt=0, le=20, description="Number of similar days"),
):
    """
    Find historically similar days based on news sentiment features.

    Returns:
        List of similar days with similarity scores.
    """
    ticker = ticker.upper()
    return get_similar_days(ticker, date, top_k=top_k)


@router.post("/news/{ticker}/refresh")
def refresh_news(
    ticker: str,
    days: int = Query(30, gt=0, le=365, description="Days of history to fetch"),
):
    """
    Force refresh news data for a ticker.

    Fetches from the configured news provider and stores in DB.

    Returns:
        Number of articles fetched and stored.
    """
    ticker = ticker.upper()
    count = fetch_and_store_news(ticker, days=days)
    return {"ticker": ticker, "articles_fetched": count}
