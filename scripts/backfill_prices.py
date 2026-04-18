"""
Backfill 2 years of daily OHLCV data from yfinance into Supabase.

Usage:
    cd /c/Users/zjg09/projects/quant-ai
    python -m scripts.backfill_prices

Tickers: the SCREENER_TICKERS list (matches frontend Screener page).
Duration: 2 years (730 days) of daily data, ending today.

Uses the configured DATABASE_URL (Supabase pooler). Upserts via
ON CONFLICT DO NOTHING so re-running is safe.
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("backfill")

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "JPM", "V", "WMT",
]


def main() -> int:
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return 1

    from app.db.prices_repo import upsert_prices

    end = date.today()
    start = end - timedelta(days=730)

    total_rows = 0
    total_failed = 0

    for ticker in TICKERS:
        try:
            logger.info(f"Fetching {ticker} ({start} to {end})...")
            hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
            if hist.empty:
                logger.warning(f"  -> no data for {ticker}")
                total_failed += 1
                continue

            rows = []
            for idx, row in hist.iterrows():
                rows.append({
                    "ticker": ticker,
                    "date": idx.date().isoformat(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                })

            upsert_prices(rows)
            logger.info(f"  -> {len(rows)} rows inserted for {ticker}")
            total_rows += len(rows)

        except Exception as e:
            logger.error(f"  -> failed {ticker}: {e}")
            total_failed += 1

    logger.info(f"")
    logger.info(f"Done. {total_rows} total rows inserted across {len(TICKERS) - total_failed}/{len(TICKERS)} tickers.")
    return 0 if total_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
