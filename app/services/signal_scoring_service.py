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
