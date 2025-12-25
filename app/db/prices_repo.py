from typing import List, Dict
from sqlalchemy import text
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

    with engine.begin() as conn:
        result = conn.execute(
            sql,
            {"ticker": ticker.upper(), "limit": limit},
        )
        rows = result.mappings().all()

    return list(rows)
