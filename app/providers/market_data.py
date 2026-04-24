"""
Market data helpers — thin wrappers over the Yahoo Finance provider.

Provides a single `fetch_ohlc(ticker, lookback_days)` function that returns
a normalized OHLC DataFrame compatible with the meta-labeling pipeline.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd


def fetch_ohlc(ticker: str, lookback_days: int = 730) -> pd.DataFrame:
    """Fetch OHLC data for *ticker* over the last *lookback_days* calendar days.

    Returns:
        DataFrame with columns: date, open, high, low, close, volume.
        Rows are sorted ascending by date and indexed 0..N-1.

    Raises:
        ValueError: if no data is returned for the ticker.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for fetch_ohlc. Install with: pip install yfinance"
        ) from exc

    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days + 10)  # buffer for weekends/holidays

    raw = yf.download(
        tickers=ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if raw is None or raw.empty:
        raise ValueError(f"No market data returned for ticker {ticker!r}")

    # Flatten MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Normalize column names to lowercase
    raw = raw.rename(columns={c: c.lower() for c in raw.columns})

    # Ensure we have the required columns
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns from Yahoo Finance response: {missing}")

    df = raw[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.reset_index()
    # Rename index column to 'date'
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Drop NaN rows
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    return df
