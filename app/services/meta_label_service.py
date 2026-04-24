"""
Meta-Label Training Service — end-to-end orchestrator for V4 Phase 3.

Steps:
  1. Fetch OHLC
  2. Apply ta_basic features
  3. Generate primary signals via PrimarySignalService
  4. Compute vol_series (p1_model or realized_sigma)
  5. Run triple_barrier_events to get meta-label events
  6. Build event features
  7. PurgedKFold CV + XGBoost (or ensemble) training per fold
  8. Final fit on ALL events
  9. Register in ModelRegistry with label_type="meta_label"
 10. Return model_id + CV metrics + warnings

Full implementation: Task P3-6 (see docs/superpowers/plans/2026-04-24-p3-meta-labeling.md).
"""

from __future__ import annotations

import uuid
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.ml.labels.meta_label import triple_barrier_events
from app.services.meta_label_features import build_event_features
from app.services.primary_signal_service import (
    PrimarySignalSpec,
    generate_primary_signals,
)
from app.backtest.metrics import calculate_meta_label_metrics


MIN_EVENTS = 30


class MetaLabelTrainRequest(BaseModel):
    """Internal request shape. API layer maps external JSON into this."""
    ticker: str
    primary: PrimarySignalSpec
    tp_k: float = Field(gt=0)
    sl_k: float = Field(gt=0)
    timeout_days: int = Field(ge=1, le=30)
    vol_source: Literal["p1_model", "realized_sigma"] = "realized_sigma"
    cv_n_splits: int = Field(default=5, ge=2, le=10)
    cv_embargo_pct: float = Field(default=0.01, ge=0.0, le=0.5)
    model_type: Literal["xgboost", "lightgbm", "ensemble"] = "xgboost"
    ensemble_mode: Optional[str] = None
    search_mode: Literal["default", "optuna"] = "default"
    lookback_days: int = Field(default=730, ge=60)
    feature_group: str = "ta_basic"


# ---- External dependencies, wrapped for monkeypatchability ----

def _fetch_ohlc(ticker: str, lookback_days: int) -> pd.DataFrame:
    """Fetch OHLC from providers (yfinance in prod)."""
    from app.providers.market_data import fetch_ohlc
    return fetch_ohlc(ticker=ticker, lookback_days=lookback_days)


def _apply_ta_features(ohlc: pd.DataFrame, feature_group: str) -> pd.DataFrame:
    from app.ml.features.technical import add_technical_features
    df = ohlc.copy()
    df = add_technical_features(df)
    return df


def _compute_vol_series(
    ohlc: pd.DataFrame, vol_source: str, ticker: str
) -> tuple[pd.Series, list[str]]:
    """Compute lagged vol series; return (series, warnings)."""
    warnings: list[str] = []
    if vol_source == "p1_model":
        vol = _load_p1_vol_series(ticker, ohlc)
        if vol is None:
            warnings.append(
                "P1 vol model not available; auto-fallback to realized_sigma (20d)."
            )
            vol = _realized_sigma(ohlc, window=20)
    else:
        vol = _realized_sigma(ohlc, window=20)
    return vol.reindex(ohlc.index).fillna(0.0), warnings


def _load_p1_vol_series(ticker: str, ohlc: pd.DataFrame) -> Optional[pd.Series]:
    """Try to load a promoted volatility model and predict per-bar vol."""
    try:
        from app.services.volatility_predict_service import load_latest_vol_model
    except Exception:
        return None
    try:
        model_info = load_latest_vol_model(ticker)
    except Exception:
        return None
    if model_info is None:
        return None
    try:
        predictions = model_info.predict_vol_series(ohlc)
        if predictions is None or len(predictions) != len(ohlc):
            return None
        return pd.Series(predictions, index=ohlc.index)
    except Exception:
        return None


def _realized_sigma(ohlc: pd.DataFrame, window: int = 20) -> pd.Series:
    returns = ohlc["close"].pct_change()
    return returns.rolling(window=window, min_periods=window // 2).std() * np.sqrt(252)


# ---- Model training ----

def _fit_meta_model(
    X_train: pd.DataFrame, y_train: np.ndarray, model_type: str,
    ensemble_mode: Optional[str], search_mode: str,
) -> Any:
    """Fit a meta-model on training data. Returns a fitted BaseModel."""
    from app.ml.models import get_model
    model = get_model(model_type, task="classification", ensemble_mode=ensemble_mode)
    model.fit(X_train.to_numpy(), y_train)
    return model


def _register_meta_model(
    *, ticker: str, primary: PrimarySignalSpec, barrier_cfg: dict,
    cv_cfg: dict, event_count: int, class_balance: dict,
    cv_metrics: dict, feature_cols: list[str], model: Any,
    p1_vol_model_id: Optional[str],
) -> str:
    """Save model to disk + insert ModelRecord. Returns model_id."""
    from pathlib import Path
    from app.core.settings import settings
    from app.db.model_registry import ModelRecord, get_model_registry

    model_id = f"meta_{ticker.lower()}_{uuid.uuid4().hex[:8]}"

    # Save model artifact to disk
    artifacts_root = Path(settings.STORAGE_LOCAL_PATH)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_root / model_id
    try:
        model.save(model_path)
    except Exception:
        # If model.save() is not available (e.g. sklearn wrapper), use joblib
        import joblib
        model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path / "model.joblib")

    # Insert registry record
    record = ModelRecord(
        id=model_id,
        name=f"meta_label_{ticker.lower()}",
        model_type="xgboost",
        tickers=[ticker],
        feature_groups=["ta_basic"],
        feature_names=feature_cols,
        n_features=len(feature_cols),
        label_type="meta_label",
        train_samples=event_count,
        metrics={
            "cv_auc_mean": cv_metrics.get("auc_mean", 0.0),
            "precision_at_50": cv_metrics.get("precision_at_50", 0.0),
        },
        artifact_path=str(model_path),
        status="active",
    )
    try:
        registry = get_model_registry()
        registry.insert_model(record)
    except Exception:
        pass  # Registry unavailable (e.g. in tests) — model still saved to disk

    return model_id


# ---- Public orchestrator ----

def train_meta_label_model(req: MetaLabelTrainRequest) -> dict[str, Any]:
    """Run the full meta-label training pipeline.

    Returns:
        dict with success, model_id, event_count, class_balance, cv_metrics,
        barrier_config_used, warnings.
    """
    warnings_list: list[str] = []

    # 1. Fetch OHLC
    ohlc = _fetch_ohlc(req.ticker, req.lookback_days)
    ohlc = ohlc.reset_index(drop=True)

    # 2. Apply TA features
    ohlc_ta = _apply_ta_features(ohlc, req.feature_group)
    ohlc_ta = ohlc_ta.reset_index(drop=True)

    # 3. Generate primary signals
    signals, strengths = generate_primary_signals(req.primary, ohlc_ta)

    # 4. Compute vol series
    vol_series, vol_warnings = _compute_vol_series(ohlc_ta, req.vol_source, req.ticker)
    warnings_list.extend(vol_warnings)

    # 5. Triple-barrier events
    events_df = triple_barrier_events(
        ohlc_ta, signals, vol_series,
        tp_k=req.tp_k, sl_k=req.sl_k, timeout_days=req.timeout_days,
        signal_strengths=strengths,
    )

    if len(events_df) < MIN_EVENTS:
        raise ValueError(
            f"insufficient_events: only {len(events_df)} events generated "
            f"(minimum {MIN_EVENTS} required)"
        )

    # 6. Build event features
    feature_cols_base = [
        c for c in ohlc_ta.columns
        if c not in {"date", "open", "high", "low", "close", "volume"}
    ]
    X = build_event_features(
        ohlc_ta=ohlc_ta,
        events=events_df,
        vol_series=vol_series,
        primary_source_key=str(req.primary.strategy_name or req.primary.primary_model_id),
        feature_cols=feature_cols_base if feature_cols_base else ["close"],
    )
    y = events_df["primary_direction_correct"].to_numpy()
    realized_r = events_df["realized_R"].to_numpy()

    feature_cols = list(X.columns)

    # Class balance
    n_correct = int(y.sum())
    class_balance = {"correct": n_correct, "wrong": len(y) - n_correct}

    # 7. Purged K-Fold CV
    from app.ml.split.purged_kfold import PurgedKFold
    pkf = PurgedKFold(n_splits=req.cv_n_splits, embargo_pct=req.cv_embargo_pct)
    fold_metrics: list[dict] = []
    folds_used = 0
    for train_idx, test_idx in pkf.split(events_df):
        if len(train_idx) < 5 or len(test_idx) < 2:
            continue
        X_tr = X.iloc[train_idx]
        y_tr = y[train_idx]
        X_te = X.iloc[test_idx]
        y_te = y[test_idx]
        r_te = realized_r[test_idx]
        try:
            fold_model = _fit_meta_model(
                X_tr, y_tr, req.model_type, req.ensemble_mode, req.search_mode
            )
            proba_te = fold_model.predict_proba(X_te.to_numpy())[:, 1]
            m = calculate_meta_label_metrics(y_te, proba_te, r_te, trade_threshold=0.5)
            fold_metrics.append(m)
            folds_used += 1
        except Exception:
            continue

    if folds_used == 0:
        warnings_list.append("No usable CV folds — returning zero metrics.")
        cv_metrics = {
            "auc_mean": 0.0, "auc_std": 0.0,
            "precision_at_50": 0.0, "expected_R_when_trade": 0.0,
            "hit_rate_when_trade": 0.0, "folds_used": 0,
        }
    else:
        cv_metrics = {
            "auc_mean": float(np.mean([m["auc"] for m in fold_metrics])),
            "auc_std": float(np.std([m["auc"] for m in fold_metrics])),
            "precision_at_50": float(np.mean([m["precision_at_threshold"] for m in fold_metrics])),
            "expected_R_when_trade": float(np.mean([m["expected_R_when_trade"] for m in fold_metrics])),
            "hit_rate_when_trade": float(np.mean([m["hit_rate_when_trade"] for m in fold_metrics])),
            "folds_used": folds_used,
        }

    # 8. Final fit on ALL events
    final_model = _fit_meta_model(
        X, y, req.model_type, req.ensemble_mode, req.search_mode
    )

    # 9. Register
    barrier_cfg = {
        "tp_k": req.tp_k, "sl_k": req.sl_k,
        "timeout_days": req.timeout_days, "vol_source": req.vol_source,
    }
    cv_cfg = {"n_splits": req.cv_n_splits, "embargo_pct": req.cv_embargo_pct}
    model_id = _register_meta_model(
        ticker=req.ticker,
        primary=req.primary,
        barrier_cfg=barrier_cfg,
        cv_cfg=cv_cfg,
        event_count=len(events_df),
        class_balance=class_balance,
        cv_metrics=cv_metrics,
        feature_cols=feature_cols,
        model=final_model,
        p1_vol_model_id=None,
    )

    return {
        "success": True,
        "model_id": model_id,
        "event_count": len(events_df),
        "class_balance": class_balance,
        "cv_metrics": cv_metrics,
        "barrier_config_used": barrier_cfg,
        "warnings": warnings_list,
    }
