"""
News Repository

Database operations for news articles and sentiment data.
Uses SQLAlchemy text() + engine pattern (same as prices_repo).

Table schema:
    CREATE TABLE IF NOT EXISTS news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        date TEXT NOT NULL,
        headline TEXT NOT NULL,
        summary TEXT,
        url TEXT,
        source TEXT,
        category TEXT,
        sentiment_score REAL,
        bullish_reason TEXT,
        bearish_reason TEXT,
        relevance_score REAL,
        created_at TEXT DEFAULT (datetime('now')),
        UNIQUE(ticker, date, url)
    );
"""

from typing import Dict, List

from sqlalchemy import text

from app.db.engine import engine


def _ensure_table() -> None:
    """Create the news table if it does not exist."""
    sql = text("""
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            headline TEXT NOT NULL,
            summary TEXT,
            url TEXT,
            source TEXT,
            category TEXT,
            sentiment_score REAL,
            bullish_reason TEXT,
            bearish_reason TEXT,
            relevance_score REAL,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(ticker, date, url)
        );
    """)
    with engine.begin() as conn:
        conn.execute(sql)


# Ensure table exists on module import
_ensure_table()


def upsert_news(rows: List[Dict]) -> None:
    """
    Insert news rows into the database.
    Ignores duplicates based on (ticker, date, url).

    Args:
        rows: List of dicts with keys matching the news table columns.
    """
    if not rows:
        return

    sql = text("""
        INSERT INTO news (
            ticker, date, headline, summary, url, source,
            category, sentiment_score, bullish_reason, bearish_reason,
            relevance_score
        )
        VALUES (
            :ticker, :date, :headline, :summary, :url, :source,
            :category, :sentiment_score, :bullish_reason, :bearish_reason,
            :relevance_score
        )
        ON CONFLICT (ticker, date, url) DO NOTHING;
    """)

    with engine.begin() as conn:
        conn.execute(sql, rows)


def get_news(
    ticker: str,
    date: str | None = None,
    limit: int = 50,
    category: str | None = None,
) -> List[Dict]:
    """
    Fetch news articles by ticker with optional filters.

    Args:
        ticker: Stock ticker symbol.
        date: Optional date filter (YYYY-MM-DD).
        limit: Maximum rows to return.
        category: Optional category filter.

    Returns:
        List of news record dicts.
    """
    conditions = ["ticker = :ticker"]
    params: dict = {"ticker": ticker.upper(), "limit": limit}

    if date:
        conditions.append("date = :date")
        params["date"] = date

    if category:
        conditions.append("category = :category")
        params["category"] = category

    where_clause = " AND ".join(conditions)

    sql = text(f"""
        SELECT ticker, date, headline, summary, url, source,
               category, sentiment_score, bullish_reason, bearish_reason,
               relevance_score, created_at
        FROM news
        WHERE {where_clause}
        ORDER BY date DESC
        LIMIT :limit
    """)

    with engine.begin() as conn:
        result = conn.execute(sql, params)
        rows = result.mappings().all()

    return [dict(r) for r in rows]


def get_news_by_date_range(
    ticker: str,
    start_date: str,
    end_date: str,
) -> List[Dict]:
    """
    Fetch news articles within a date range.

    Args:
        ticker: Stock ticker symbol.
        start_date: Start date (YYYY-MM-DD), inclusive.
        end_date: End date (YYYY-MM-DD), inclusive.

    Returns:
        List of news record dicts.
    """
    sql = text("""
        SELECT ticker, date, headline, summary, url, source,
               category, sentiment_score, bullish_reason, bearish_reason,
               relevance_score, created_at
        FROM news
        WHERE ticker = :ticker AND date >= :start_date AND date <= :end_date
        ORDER BY date DESC
    """)

    with engine.begin() as conn:
        result = conn.execute(
            sql,
            {
                "ticker": ticker.upper(),
                "start_date": start_date,
                "end_date": end_date,
            },
        )
        rows = result.mappings().all()

    return [dict(r) for r in rows]


def get_news_categories(ticker: str) -> List[Dict]:
    """
    Get distinct categories and their counts for a ticker.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        List of dicts with 'category' and 'count' keys.
    """
    sql = text("""
        SELECT category, COUNT(*) as count
        FROM news
        WHERE ticker = :ticker AND category IS NOT NULL
        GROUP BY category
        ORDER BY count DESC
    """)

    with engine.begin() as conn:
        result = conn.execute(sql, {"ticker": ticker.upper()})
        rows = result.mappings().all()

    return [dict(r) for r in rows]
