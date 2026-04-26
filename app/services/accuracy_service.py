"""
Accuracy Service — V4 P5 G1

Lazily resolves prediction_log rows whose horizon has passed by fetching
actual market data, then aggregates 30-day stats. No cron — resolution
happens on demand when /models/{id}/accuracy is called.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import sqrt
from typing import Any
import logging

import numpy as np
import pandas as pd

from app.db import prediction_log as _pred_log_module
from app.db.prediction_log import PredictionLogRecord

logger = logging.getLogger(__name__)


def _get_repo():
    """Indirected for monkeypatching in tests."""
    return _pred_log_module.get_prediction_log_repo()


def _fetch_ohlc_slice(ticker: str, start: datetime, end: datetime) -> pd.DataFrame | None:
    """Fetch OHLC slice [start, end] from market provider. Returns None on failure."""
    try:
        from app.providers import get_market_provider
        provider = get_market_provider()
        df = provider.fetch(ticker=ticker, start_date=start.date(), end_date=end.date())
        return df
    except Exception as e:
        logger.warning("fetch_ohlc_slice failed for %s [%s..%s]: %s", ticker, start, end, e)
        return None


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with date column as tz-naive date strings (date only)."""
    df2 = df.copy()
    df2["_date_only"] = pd.to_datetime(df2["date"]).dt.normalize().dt.tz_localize(None)
    return df2


def _close_at(df: pd.DataFrame, target_dt: datetime) -> float | None:
    if df is None or df.empty:
        return None
    df2 = _normalize_dates(df)
    target = pd.Timestamp(target_dt.date())
    mask = df2["_date_only"] == target
    if mask.any():
        v = float(df2.loc[mask, "close"].iloc[0])
        return v if np.isfinite(v) else None
    # Find the closest prior bar
    valid = df2[df2["_date_only"] <= target]
    if valid.empty:
        return None
    v = float(valid.iloc[-1]["close"])
    return v if np.isfinite(v) else None


def _rolling_vol_at(df: pd.DataFrame, target_dt: datetime, window: int = 20) -> float:
    if df is None or df.empty:
        return 0.02
    df2 = _normalize_dates(df)
    target = pd.Timestamp(target_dt.date())
    prior = df2[df2["_date_only"] <= target].tail(window + 1)
    if len(prior) < 5:
        return 0.02
    rets = prior["close"].pct_change().dropna()
    sigma = float(rets.std()) * sqrt(252) if len(rets) > 1 else 0.02
    return max(sigma, 1e-6)


def _realized_vol(df: pd.DataFrame, t0: datetime, t1: datetime) -> float | None:
    if df is None or df.empty:
        return None
    df2 = _normalize_dates(df)
    t0_ts = pd.Timestamp(t0.date())
    t1_ts = pd.Timestamp(t1.date())
    window = df2[(df2["_date_only"] >= t0_ts) & (df2["_date_only"] <= t1_ts)]
    if len(window) < 2:
        return None
    rets = window["close"].pct_change().dropna()
    if len(rets) < 1:
        return None
    return float(rets.std()) * sqrt(252)


def resolve_pending(model_id: str, limit: int = 100) -> dict[str, int]:
    """Resolve all unresolved predictions for `model_id` whose horizon has passed."""
    repo = _get_repo()
    pending = repo.list_unresolved(model_id, limit=limit)
    checked, newly_resolved, errors = 0, 0, 0

    for rec in pending:
        checked += 1
        try:
            slice_start = rec.predicted_at - timedelta(days=30)
            slice_end = rec.resolve_at + timedelta(days=2)
            df = _fetch_ohlc_slice(rec.ticker, slice_start, slice_end)
            if df is None or df.empty:
                errors += 1
                continue

            close_predict = _close_at(df, rec.predicted_at)
            close_resolve = _close_at(df, rec.resolve_at)
            if close_predict is None or close_resolve is None:
                errors += 1
                continue

            actual_return = (close_resolve - close_predict) / close_predict

            if rec.label_type in ("direction", "meta_label"):
                vol = _rolling_vol_at(df, rec.predicted_at)
                signal = rec.predicted_signal or 0
                is_correct = (signal == 1 and actual_return > 0) or (signal == -1 and actual_return < 0)
                realized_R = signal * actual_return / vol if vol > 0 else 0.0
                repo.update_resolution(
                    rec.id,
                    actual_value=close_resolve,
                    actual_return=actual_return,
                    is_correct=bool(is_correct),
                    realized_R=float(realized_R),
                )
            else:  # volatility
                rv = _realized_vol(df, rec.predicted_at, rec.resolve_at)
                if rv is None:
                    errors += 1
                    continue
                repo.update_resolution(
                    rec.id,
                    actual_value=float(rv),
                    actual_return=actual_return,
                )
            newly_resolved += 1
        except Exception as e:
            logger.warning("resolve failed for %s: %s", rec.id, e)
            errors += 1

    return {"checked": checked, "newly_resolved": newly_resolved, "errors": errors}


def aggregate(model_id: str, window_days: int = 30) -> dict[str, Any]:
    """Aggregate accuracy stats for `model_id` over the last `window_days`."""
    repo = _get_repo()
    since = datetime.now(timezone.utc) - timedelta(days=window_days)
    rows = repo.list_by_model_id(model_id, since=since, limit=500)
    resolved = [r for r in rows if r.resolved_at]
    pending = [r for r in rows if not r.resolved_at]

    label_type = rows[0].label_type if rows else None

    stats: dict[str, Any] = {
        "total_predictions": len(rows),
        "resolved": len(resolved),
        "pending": len(pending),
        "hit_rate": None,
        "avg_realized_R": None,
        "best_R": None,
        "worst_R": None,
        "mae": None,
        "rmse": None,
    }

    if not resolved:
        return stats

    if label_type in ("direction", "meta_label"):
        correct = [r for r in resolved if r.is_correct]
        rs = [r.realized_R for r in resolved if r.realized_R is not None]
        stats["hit_rate"] = len(correct) / len(resolved)
        if rs:
            stats["avg_realized_R"] = float(np.mean(rs))
            stats["best_R"] = float(max(rs))
            stats["worst_R"] = float(min(rs))
    elif label_type == "volatility":
        diffs = [
            r.actual_value - r.predicted_value
            for r in resolved
            if r.actual_value is not None
        ]
        if diffs:
            stats["mae"] = float(np.mean([abs(d) for d in diffs]))
            stats["rmse"] = float(np.sqrt(np.mean([d * d for d in diffs])))

    return stats


def by_ticker(model_id: str, window_days: int = 30) -> list[dict[str, Any]]:
    repo = _get_repo()
    since = datetime.now(timezone.utc) - timedelta(days=window_days)
    rows = repo.list_by_model_id(model_id, since=since, limit=500)
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        slot = out.setdefault(r.ticker, {
            "ticker": r.ticker, "total": 0, "resolved": 0,
            "hits": 0, "rs": [],
        })
        slot["total"] += 1
        if r.resolved_at:
            slot["resolved"] += 1
            if r.is_correct:
                slot["hits"] += 1
            if r.realized_R is not None:
                slot["rs"].append(r.realized_R)
    return [
        {
            "ticker": s["ticker"], "total": s["total"], "resolved": s["resolved"],
            "hit_rate": (s["hits"] / s["resolved"]) if s["resolved"] else None,
            "avg_R": (float(np.mean(s["rs"])) if s["rs"] else None),
        }
        for s in out.values()
    ]


def last_predictions(model_id: str, limit: int = 20) -> list[dict[str, Any]]:
    repo = _get_repo()
    rows = repo.list_by_model_id(model_id, limit=limit)
    return [r.model_dump(mode="json") for r in rows[:limit]]
