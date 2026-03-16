"""
News Service

Business logic layer for news data: fetching, storing, querying,
and sentiment analysis.
"""

import logging
from collections import Counter
from datetime import date, timedelta

import numpy as np

from app.core.settings import settings
from app.db.news_repo import (
    get_news,
    get_news_by_date_range,
    get_news_categories,
    upsert_news,
)
from app.providers.factory import get_news_provider

logger = logging.getLogger(__name__)


def fetch_and_store_news(ticker: str, days: int = 30) -> int:
    """
    Fetch news from provider and store in DB.

    Args:
        ticker: Stock ticker symbol.
        days: Number of days of history to fetch.

    Returns:
        Number of articles stored.
    """
    ticker = ticker.upper()
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    try:
        provider = get_news_provider(settings.NEWS_PROVIDER)
    except (ValueError, Exception) as e:
        logger.warning(f"Failed to get provider '{settings.NEWS_PROVIDER}': {e}, falling back to mock")
        provider = get_news_provider("mock")

    logger.info(f"Fetching news for {ticker} from {provider.provider_name}")

    try:
        df = provider.fetch(
            ticker,
            start_date=start_date,
            end_date=end_date,
            limit=100,
        )
    except ValueError as e:
        logger.warning(f"Provider fetch failed: {e}, falling back to mock")
        provider = get_news_provider("mock")
        df = provider.fetch(ticker, start_date=start_date, end_date=end_date, limit=100)

    if df.empty:
        logger.info(f"No news fetched for {ticker}")
        return 0

    records = df.to_dict(orient="records")
    upsert_news(records)
    logger.info(f"Stored {len(records)} news articles for {ticker}")
    return len(records)


def get_news_for_ticker(
    ticker: str,
    date: str | None = None,
    limit: int = 50,
    category: str | None = None,
) -> list[dict]:
    """
    Get news for a ticker. Checks DB first; fetches from provider if empty.

    Args:
        ticker: Stock ticker symbol.
        date: Optional date filter (YYYY-MM-DD).
        limit: Maximum articles to return.
        category: Optional category filter.

    Returns:
        List of news record dicts.
    """
    ticker = ticker.upper()

    rows = get_news(ticker, date=date, limit=limit, category=category)
    if rows:
        return rows

    # DB empty — try fetching
    logger.info(f"No cached news for {ticker}, fetching from provider")
    count = fetch_and_store_news(ticker)
    if count == 0:
        return []

    return get_news(ticker, date=date, limit=limit, category=category)


def get_news_sentiment_summary(ticker: str, target_date: str) -> dict:
    """
    Get sentiment summary for a ticker on a specific date.

    Args:
        ticker: Stock ticker symbol.
        target_date: Date string (YYYY-MM-DD).

    Returns:
        Dict with avg_sentiment, news_count, category_distribution, etc.
    """
    ticker = ticker.upper()
    rows = get_news(ticker, date=target_date, limit=500)

    if not rows:
        return {
            "ticker": ticker,
            "date": target_date,
            "news_count": 0,
            "avg_sentiment": None,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "category_distribution": {},
        }

    scores = [
        r["sentiment_score"]
        for r in rows
        if r.get("sentiment_score") is not None
    ]

    positive = sum(1 for s in scores if s > 0.1)
    negative = sum(1 for s in scores if s < -0.1)
    neutral = len(scores) - positive - negative

    categories = Counter(r.get("category", "other") for r in rows)

    return {
        "ticker": ticker,
        "date": target_date,
        "news_count": len(rows),
        "avg_sentiment": round(float(np.mean(scores)), 4) if scores else None,
        "positive_count": positive,
        "negative_count": negative,
        "neutral_count": neutral,
        "category_distribution": dict(categories),
    }


def get_similar_days(
    ticker: str,
    target_date: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Find historically similar days based on news sentiment features.

    Uses cosine similarity on a feature vector:
    [news_count, avg_sentiment, positive_ratio, negative_ratio].

    Args:
        ticker: Stock ticker symbol.
        target_date: Reference date (YYYY-MM-DD).
        top_k: Number of similar days to return.

    Returns:
        List of dicts with date and similarity score.
    """
    ticker = ticker.upper()

    # Get target day's features
    target_summary = get_news_sentiment_summary(ticker, target_date)
    if target_summary["news_count"] == 0:
        return []

    target_vec = _build_feature_vector(target_summary)

    # Get all historical news within a wide range
    end = target_date
    start_dt = date.fromisoformat(target_date) - timedelta(days=365)
    start = start_dt.isoformat()

    all_rows = get_news_by_date_range(ticker, start, end)
    if not all_rows:
        return []

    # Group by date
    by_date: dict[str, list] = {}
    for r in all_rows:
        d = r["date"]
        if d == target_date:
            continue
        by_date.setdefault(d, []).append(r)

    # Compute similarity for each date
    similarities = []
    for d, day_rows in by_date.items():
        day_summary = _summarize_rows(ticker, d, day_rows)
        day_vec = _build_feature_vector(day_summary)
        sim = _cosine_similarity(target_vec, day_vec)
        similarities.append({"date": d, "similarity": round(sim, 4), **day_summary})

    # Sort by similarity descending
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_k]


def _build_feature_vector(summary: dict) -> np.ndarray:
    """Build a feature vector from a sentiment summary."""
    count = summary.get("news_count", 0)
    avg = summary.get("avg_sentiment") or 0.0
    pos_ratio = summary.get("positive_count", 0) / max(count, 1)
    neg_ratio = summary.get("negative_count", 0) / max(count, 1)
    return np.array([count, avg, pos_ratio, neg_ratio], dtype=float)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _summarize_rows(ticker: str, d: str, rows: list[dict]) -> dict:
    """Build a summary dict from raw news rows for a single date."""
    scores = [
        r["sentiment_score"]
        for r in rows
        if r.get("sentiment_score") is not None
    ]
    positive = sum(1 for s in scores if s > 0.1)
    negative = sum(1 for s in scores if s < -0.1)
    neutral = len(scores) - positive - negative
    categories = Counter(r.get("category", "other") for r in rows)

    return {
        "ticker": ticker,
        "date": d,
        "news_count": len(rows),
        "avg_sentiment": round(float(np.mean(scores)), 4) if scores else None,
        "positive_count": positive,
        "negative_count": negative,
        "neutral_count": neutral,
        "category_distribution": dict(categories),
    }
