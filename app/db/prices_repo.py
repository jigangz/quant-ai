from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from app.db.engine import engine


def upsert_prices(rows: List[Dict]) -> None:
    """
    Insert OHLCV rows into prices table.
    Ignore duplicates based on (ticker, date).
    """
    if not rows:
        return

    sql = text("""
        insert into prices (ticker, date, open, high, low, close, volume)
        values (:ticker, :date, :open, :high, :low, :close, :volume)
        on conflict (ticker, date)
        do nothing;
    """)

    with engine.begin() as conn:
        conn.execute(sql, rows)


def get_prices(
    ticker: str,
    limit: int = 30,
) -> List[Dict]:
    """
    Fetch recent OHLCV data by ticker.
    """
    sql = text("""
        select ticker, date, open, high, low, close, volume
        from prices
        where ticker = :ticker
        order by date desc
        limit :limit
    """)

    try:
        with engine.begin() as conn:
            result = conn.execute(
                sql,
                {"ticker": ticker.upper(), "limit": limit},
            )
            rows = result.mappings().all()
    except OperationalError:
        # Table may not exist in test/dev environments — treat as no data
        return []

    return list(rows)


def get_prices_batch(
    tickers: List[str],
    since_days: int = 520,
) -> Dict[str, pd.DataFrame]:
    """Fetch recent OHLCV for many tickers in ONE query (V5 Phase D ranking).

    Avoids N round-trips when scoring the whole universe. Returns a dict mapping
    each ticker to its rows as an ascending-by-date DataFrame; tickers with no
    rows in the window are absent. Empty/error → {}.
    """
    if not tickers:
        return {}

    since = datetime.now(timezone.utc).date() - timedelta(days=since_days)
    sql = text("""
        select ticker, date, open, high, low, close, volume
        from prices
        where ticker = any(:tickers) and date >= :since
        order by ticker, date
    """)
    try:
        with engine.begin() as conn:
            rows = conn.execute(
                sql,
                {"tickers": [t.upper() for t in tickers], "since": since},
            ).mappings().all()
    except OperationalError:
        return {}

    grouped: Dict[str, list] = {}
    for r in rows:
        grouped.setdefault(r["ticker"], []).append(dict(r))

    return {
        t: pd.DataFrame(v).sort_values("date").reset_index(drop=True)
        for t, v in grouped.items()
    }


def get_prices_df(
    ticker: str,
    limit: int = 500,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data as a DataFrame for strategy calculations.
    
    Args:
        ticker: Stock ticker symbol
        limit: Maximum rows to return
        
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
        Returns None if no data found
    """
    rows = get_prices(ticker, limit)
    
    if not rows:
        return None
    
    df = pd.DataFrame(rows)
    # Sort by date ascending for time-series analysis
    df = df.sort_values("date").reset_index(drop=True)
    
    return df
