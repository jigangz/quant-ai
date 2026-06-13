"""Backfill daily OHLCV from yfinance into Supabase.

Usage:
    python -m scripts.backfill_prices                      # legacy 10-ticker list
    python -m scripts.backfill_prices --universe sp500     # full S&P 500 (point-in-time backfill set + semis)
    python -m scripts.backfill_prices --universe sp500 --limit 120   # first 120 (MVP)

V5: batch-downloads via yf.download (fewer rate-limit hits than per-ticker
loops), retries with backoff, and pulls the *point-in-time backfill universe*
(current S&P 500 + names removed since the start date) so cross-sectional
training has the dropped tickers too — survivorship-bias fix. Idempotent
(ON CONFLICT DO NOTHING).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("backfill")

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "JPM", "V", "WMT",
]
# Important semis that aren't in the S&P 500 (non-US), per Harry's watchlist.
EXTRA_TICKERS = ["ASML", "TSM", "LITE"]


def resolve_universe(name: str, since: str, limit: int | None) -> list[str]:
    if name == "sp500":
        from app.data.universe import backfill_universe

        tickers = backfill_universe(since)
        for t in EXTRA_TICKERS:
            if t not in tickers:
                tickers.append(t)
        tickers = sorted(set(tickers))
    else:
        tickers = list(DEFAULT_TICKERS)
    return tickers[:limit] if limit else tickers


def _fetch_batch(tickers: list[str], start: date, end: date, retries: int = 3):
    import yfinance as yf

    for attempt in range(retries):
        try:
            df = yf.download(
                tickers, start=start, end=end, interval="1d",
                group_by="ticker", auto_adjust=True, threads=True, progress=False,
            )
            if df is not None and not df.empty:
                return df
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  batch attempt {attempt + 1} failed: {e}")
        time.sleep(5 * (attempt + 1))  # backoff
    return None


def _rows_for(df: pd.DataFrame, ticker: str, multi: bool) -> list[dict]:
    try:
        sub = df[ticker] if multi else df
    except KeyError:
        return []
    rows = []
    for idx, row in sub.iterrows():
        close = row.get("Close")
        if close is None or pd.isna(close):
            continue
        rows.append({
            "ticker": ticker,
            "date": idx.date().isoformat(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(close),
            "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="default", choices=["default", "sp500"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--batch", type=int, default=40, help="tickers per yf.download call")
    args = ap.parse_args()

    try:
        import yfinance  # noqa: F401
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return 1
    from app.db.prices_repo import upsert_prices

    end = date.today()
    start = end - timedelta(days=args.days)
    tickers = resolve_universe(args.universe, since=start.isoformat(), limit=args.limit)
    logger.info(f"Universe '{args.universe}': {len(tickers)} tickers, {start} → {end}")

    total_rows = 0
    ok = 0
    failed: list[str] = []
    for i in range(0, len(tickers), args.batch):
        batch = tickers[i : i + args.batch]
        logger.info(f"Batch {i // args.batch + 1}: {len(batch)} tickers ({batch[0]}..{batch[-1]})")
        df = _fetch_batch(batch, start, end)
        if df is None:
            logger.warning(f"  batch failed entirely: {batch}")
            failed.extend(batch)
            continue
        multi = isinstance(df.columns, pd.MultiIndex)
        for t in batch:
            rows = _rows_for(df, t, multi)
            if not rows:
                failed.append(t)
                continue
            upsert_prices(rows)
            total_rows += len(rows)
            ok += 1
        time.sleep(2)  # be gentle between batches

    logger.info("")
    logger.info(f"Done. {total_rows} rows across {ok}/{len(tickers)} tickers.")
    if failed:
        logger.info(f"No data for {len(failed)}: {failed[:15]}{'...' if len(failed) > 15 else ''}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
