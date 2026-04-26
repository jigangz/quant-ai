"""
Signal Scoring Service — inference-time orchestrator for meta-labeling (V4 Phase 3).

Three modes:
  A) Explicit signal:   {ticker, signal, timestamp, meta_model_id}
  B) Auto-trigger:      {ticker, strategy_name|primary_model_id, meta_model_id}
     → runs primary on latest OHLC → if triggered, proceed with that signal
  C) Fallback:          explicit wins if both given; otherwise B path

Returns { triggered, signal, reliability_score, expected_R,
          recommended_action, sizing_hint, meta_model, timestamp }.

Full implementation: Task P3-7 (see docs/superpowers/plans/2026-04-24-p3-meta-labeling.md).
This module is importable now so that Paper Trading (P3-10) can monkeypatch it in tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.services.primary_signal_service import (
    PrimarySignalSpec,
    generate_primary_signals,
)
from app.services.meta_label_features import build_event_features


class SignalScoreRequest(BaseModel):
    ticker: str
    meta_model_id: str
    signal: Optional[int] = None      # -1 or 1; if given → Mode A
    timestamp: Optional[str] = None
    strategy_name: Optional[str] = None
    strategy_params: Optional[dict] = None
    primary_model_id: Optional[str] = None


# ---- Wrapped externals (monkeypatchable) ----

def _load_meta_model(model_id: str) -> dict[str, Any]:
    from app.services.model_cache import get_model_cache
    cache = get_model_cache()
    info = cache.load(model_id)
    if info is None:
        raise ValueError(f"meta_model_not_found:{model_id}")
    return {
        "model": info.model, "metadata": info.metadata,
        "extras": getattr(info, "extras", {}),
    }


def _fetch_ohlc_ta(ticker: str, lookback: int, feature_group: str) -> pd.DataFrame:
    from app.services.meta_label_service import _fetch_ohlc, _apply_ta_features
    ohlc = _fetch_ohlc(ticker, lookback)
    return _apply_ta_features(ohlc, feature_group)


# ---- Orchestrator ----

def score_signal(req: SignalScoreRequest) -> dict[str, Any]:
    """Score a signal using a trained meta-label model.

    Full implementation per Task 7 of the P3 plan. For now returns a skeleton
    response until the full service is wired in P3-7.
    """
    record = _load_meta_model(req.meta_model_id)
    meta_cfg = record.get("extras", {}).get("meta_label", {})
    primary_cfg = meta_cfg.get("primary", {})
    barrier_cfg = meta_cfg.get("barrier", {})
    feature_group = record.get("metadata", {}).get("feature_group", "ta_basic")
    feature_set = meta_cfg.get("feature_set", [])

    ohlc_ta = _fetch_ohlc_ta(req.ticker, 300, feature_group)

    # Resolve mode
    if req.signal is not None:
        # Mode A: explicit signal
        target_ts = (
            pd.Timestamp(req.timestamp) if req.timestamp
            else pd.Timestamp(ohlc_ta["date"].iloc[-1])
        )
        signal = int(req.signal)
        signal_strength = 1.0
    else:
        # Mode B / C: auto-trigger via primary
        source = primary_cfg.get("source", "strategy")
        if req.strategy_name:
            spec = PrimarySignalSpec(source="strategy", strategy_name=req.strategy_name,
                                     strategy_params=req.strategy_params or {})
        elif req.primary_model_id:
            spec = PrimarySignalSpec(source="model", primary_model_id=req.primary_model_id)
        else:
            spec = PrimarySignalSpec(**primary_cfg)
        sigs, strs = generate_primary_signals(spec, ohlc_ta)
        last_nonzero = sigs[sigs != 0]
        if len(last_nonzero) == 0:
            return {
                "triggered": False, "signal": 0, "reason": "primary_silent",
                "reliability_score": 0.0, "expected_R": 0.0,
                "recommended_action": "skip",
                "sizing_hint": {"half_kelly_fraction": 0.0, "raw_kelly": 0.0, "cap": 0.25},
                "meta_model": {"id": req.meta_model_id},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        last_idx = last_nonzero.index[-1]
        signal = int(last_nonzero.iloc[-1])
        signal_strength = float(strs.iloc[last_idx])
        target_ts = pd.Timestamp(ohlc_ta["date"].iloc[last_idx])

    # Build features at target_ts − 1 bar
    ohlc_dates = pd.to_datetime(ohlc_ta["date"])
    date_idx_map = {pd.Timestamp(d): i for i, d in enumerate(ohlc_dates)}
    row_idx = date_idx_map.get(target_ts)
    if row_idx is None or row_idx == 0:
        feat_row = {c: 0.0 for c in feature_set}
    else:
        lag = row_idx - 1
        feat_row = {c: float(ohlc_ta[c].iloc[lag]) for c in feature_set if c in ohlc_ta.columns}

    feat_row.setdefault("signal_time_vol", 0.0)
    feat_row.setdefault("signal_strength", signal_strength)
    feat_row.setdefault("time_since_last_signal", float(len(ohlc_ta)))

    meta_model = record["model"]
    ordered_cols = feature_set if feature_set else list(feat_row.keys())
    X = np.array([[feat_row.get(c, 0.0) for c in ordered_cols]])
    proba = float(meta_model.predict_proba(X)[0, 1])

    # Scoring thresholds
    if proba >= 0.65:
        action = "trade"
    elif proba >= 0.45:
        action = "reduce"
    else:
        action = "skip"

    # Half-Kelly sizing
    cap = 0.25
    edge = max(0.0, proba - 0.5)
    raw_kelly = edge / max(1.0 - proba, 0.01)
    half_kelly = min(raw_kelly / 2, cap)

    cv_auc = meta_cfg.get("cv", {}).get("metrics", {}).get("auc_mean", 0.0)

    return {
        "triggered": True,
        "signal": signal,
        "reliability_score": proba,
        "expected_R": proba * barrier_cfg.get("tp_k", 2.0) - (1 - proba) * barrier_cfg.get("sl_k", 1.0),
        "recommended_action": action,
        "sizing_hint": {"half_kelly_fraction": half_kelly, "raw_kelly": raw_kelly, "cap": cap},
        "meta_model": {
            "id": req.meta_model_id,
            "primary_source": f"{primary_cfg.get('source', 'strategy')}:{primary_cfg.get('strategy_name', '')}",
            "cv_auc": cv_auc,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =====================================
# Coverage aggregation (P4-1)
# =====================================

KNOWN_STRATEGIES = {"ma_cross", "rsi_strategy", "bollinger_breakout", "sentiment_driven"}


def _list_meta_records() -> list[dict]:
    """Return all registered meta-label model records.

    Reads from the actual ModelRegistry (LocalModelRegistry or
    SupabaseModelRegistry) — the same place `_register_meta_model` writes
    to. Each entry is shaped {model_id, metadata{ticker, label_type, ...},
    extras{meta_label: {...}}, metrics} so coverage can filter by primary
    strategy. Wrapped as a function so contract tests can monkeypatch.
    """
    try:
        from app.db.model_registry import get_model_registry
        registry = get_model_registry()
        records = registry.list_models(label_type="meta_label", limit=200)
    except Exception:
        return []

    out: list[dict] = []
    for rec in records:
        ticker = rec.tickers[0] if rec.tickers else None
        out.append({
            "model_id": rec.id,
            "metadata": {
                "ticker": ticker,
                "label_type": rec.label_type,
                "model_type": rec.model_type,
            },
            "extras": rec.extras or {},
            "metrics": rec.metrics or {},
        })
    return out


# V4 P5: prediction_log write helper for meta-label Mode A path.
def _write_prediction_log(
    *,
    ticker: str,
    model_id: str,
    model_type: str,
    horizon_days: int,
    predicted_value: float,
    predicted_signal: int,
    primary_source: str,
    expected_R: float,
    feature_group: str,
) -> None:
    try:
        from datetime import datetime, timedelta, timezone
        from app.db.prediction_log import PredictionLogRecord, get_prediction_log_repo

        repo = get_prediction_log_repo()
        now = datetime.now(timezone.utc)
        repo.insert(
            PredictionLogRecord(
                model_id=model_id,
                ticker=ticker,
                label_type="meta_label",
                horizon_days=horizon_days,
                predicted_value=float(predicted_value),
                predicted_signal=predicted_signal,
                predicted_extras={
                    "feature_group": feature_group,
                    "model_type": model_type,
                    "primary_source": primary_source,
                    "expected_R": float(expected_R),
                },
                resolve_at=now + timedelta(days=horizon_days),
            )
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "prediction_log write failed (non-blocking): %s", e
        )


def compute_coverage(strategy_name: str) -> dict:
    """Aggregate meta-label coverage for a given primary strategy."""
    if strategy_name not in KNOWN_STRATEGIES:
        raise ValueError(f"strategy_not_found:{strategy_name}")

    records = _list_meta_records()
    models = []
    for r in records:
        try:
            extras = r.get("extras", {}) or {}
            meta_cfg = extras.get("meta_label") or {}
            primary_cfg = meta_cfg.get("primary") or {}
            if primary_cfg.get("source") != "strategy":
                continue
            if primary_cfg.get("strategy_name") != strategy_name:
                continue
            # Prefer extras.meta_label.cv.metrics.auc_mean if present;
            # fall back to top-level metrics["cv_auc_mean"] (older records
            # written before extras was added).
            auc = (
                meta_cfg.get("cv", {})
                .get("metrics", {})
                .get("auc_mean")
            )
            if auc is None:
                auc = r.get("metrics", {}).get("cv_auc_mean")
            if auc is None:
                continue
            ticker = r.get("metadata", {}).get("ticker") or "UNKNOWN"
            models.append({
                "model_id": r["model_id"],
                "ticker": ticker,
                "auc_mean": float(auc),
                "event_count": int(meta_cfg.get("event_count", 0)),
            })
        except (KeyError, TypeError, AttributeError):
            continue

    if not models:
        return {
            "strategy_name": strategy_name,
            "count": 0, "max_auc": None, "avg_auc": None,
            "tickers": [], "models": [],
        }

    aucs = [m["auc_mean"] for m in models]
    tickers = sorted({m["ticker"] for m in models})
    return {
        "strategy_name": strategy_name,
        "count": len(models),
        "max_auc": max(aucs),
        "avg_auc": sum(aucs) / len(aucs),
        "tickers": tickers,
        "models": models,
    }
