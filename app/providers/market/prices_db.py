from __future__ import annotations

"""Prices-DB market provider — V5 Phase C.

Cross-sectional training must read the SAME data prod serves and the SAME
point-in-time universe that was backfilled into the `prices` table (incl.
delisted names yahoo no longer returns). This provider implements the standard
MarketProvider.fetch contract against `app.db.prices_repo` so DatasetBuilder can
build the cross-sectional panel from the DB instead of re-downloading yahoo.
"""

from datetime import date
from typing import Literal

import pandas as pd

from app.providers.base import MarketProvider

# Effectively "all history" — the prices table holds ~2y daily per ticker.
_ALL = 100_000


class PricesDBProvider(MarketProvider):
    """Reads OHLCV from the prices table via prices_repo."""

    @property
    def provider_name(self) -> str:
        return "db"

    def fetch(
        self,
        ticker: str,
        start_date: date | None = None,
        end_date: date | None = None,
        interval: str = "1d",
        **kwargs,
    ) -> pd.DataFrame:
        # Local import keeps the engine off the import path (and monkeypatchable).
        from app.db.prices_repo import get_prices_df

        df = get_prices_df(ticker, limit=_ALL)
        if df is None or df.empty:
            return pd.DataFrame()

        required = {"date", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"PricesDBProvider: missing columns {missing} for {ticker}")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        if start_date is not None:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df["date"] <= pd.Timestamp(end_date)]
        return df.sort_values("date").reset_index(drop=True)
