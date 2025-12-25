from typing import Optional
import pandas as pd
import yfinance as yf


def fetch_ohlcv(
    ticker: str,
    period: str = "1mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.

    Args:
        ticker: Stock ticker, e.g. "AAPL"
        period: Data range, e.g. "5d", "1mo", "3mo", "1y", "max"
        interval: Data interval, e.g. "1d", "1h"

    Returns:
        DataFrame with columns:
        [ticker, date, open, high, low, close, volume]
    """
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    df["ticker"] = ticker.upper()

    return df[["ticker", "date", "open", "high", "low", "close", "volume"]]

