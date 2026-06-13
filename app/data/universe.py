"""Point-in-time S&P 500 universe — survivorship-bias fix (V5).

Naively using *today's* S&P 500 members to label/backtest the past leaks
survivorship bias: stocks that were dropped (bankruptcy, M&A, market-cap)
vanish from the universe, so the backtest only ever sees winners.

This reconstructs membership AS OF any historical date from two free
Wikipedia tables — the current constituents + the full change log
(Effective Date / Added ticker / Removed ticker). Walking the changes that
happened *after* a target date backwards (undo each: drop the added, restore
the removed) yields the real membership on that date.

Residual limitation (documented, honest-by-design): fully delisted names may
have incomplete yfinance history. This fix removes the bias *within data
coverage*, which is the standard free-data approach.
"""

from __future__ import annotations

import datetime as dt
from io import StringIO

import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS = {"User-Agent": "Mozilla/5.0 (research; quant-ai)"}  # Wikipedia 403s the default UA


def _yf(ticker: str) -> str:
    """Wikipedia writes class shares as BRK.B; yfinance wants BRK-B."""
    return str(ticker).strip().replace(".", "-")


def fetch_wiki_tables(html: str | None = None) -> list[pd.DataFrame]:
    """Fetch the Wikipedia tables (or parse provided html, for tests)."""
    if html is None:
        import requests

        html = requests.get(WIKI_URL, headers=_HEADERS, timeout=30).text
    return pd.read_html(StringIO(html))


def current_constituents(html: str | None = None) -> list[str]:
    """Today's S&P 500 tickers (yfinance-formatted)."""
    table = fetch_wiki_tables(html)[0]
    return sorted({_yf(s) for s in table["Symbol"].astype(str)})


def parse_changes(html: str | None = None) -> pd.DataFrame:
    """Normalize the change log into columns: date, added, removed."""
    raw = fetch_wiki_tables(html)[1].copy()
    # Flatten the ('Added','Ticker') style MultiIndex to add_t / rem_t / date.
    raw.columns = ["date", "add_t", "add_s", "rem_t", "rem_s", "reason"]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"])
    return pd.DataFrame(
        {
            "date": raw["date"],
            "added": raw["add_t"].map(lambda t: _yf(t) if pd.notna(t) else None),
            "removed": raw["rem_t"].map(lambda t: _yf(t) if pd.notna(t) else None),
        }
    )


def reconstruct_members(current: set[str], changes: pd.DataFrame, target: pd.Timestamp) -> set[str]:
    """Pure logic (testable, no network): membership on `target`.

    For every change effective AFTER target, undo it — the added ticker
    wasn't a member yet (remove it), the removed ticker still was (add it).
    """
    members = set(current)
    for _, row in changes.iterrows():
        if row["date"] > target:
            if row["added"]:
                members.discard(row["added"])
            if row["removed"]:
                members.add(row["removed"])
    return members


def members_on(target: dt.date | str, html: str | None = None) -> list[str]:
    """S&P 500 membership as of `target` (yfinance-formatted, sorted)."""
    ts = pd.Timestamp(target)
    return sorted(reconstruct_members(set(current_constituents(html)), parse_changes(html), ts))


def backfill_universe(since: dt.date | str, html: str | None = None) -> list[str]:
    """Every ticker that was a member at ANY point since `since` — the set
    to download prices for (current members + everyone removed after `since`),
    so the point-in-time universe always has data behind it."""
    ts = pd.Timestamp(since)
    members = set(current_constituents(html))
    ch = parse_changes(html)
    for _, row in ch.iterrows():
        if row["date"] >= ts and row["removed"]:
            members.add(row["removed"])
    return sorted(members)
