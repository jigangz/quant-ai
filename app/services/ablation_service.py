"""
Ablation Service — V4 P5 FE-ENH-4

Trains 6 models (3 targets × 2 feature sets) using existing training
infrastructure and returns a delta matrix. Default params for fair
comparison — Optuna would obscure feature contribution.
"""

from __future__ import annotations

import time
from typing import Any
import logging

logger = logging.getLogger(__name__)

KNOWN_FEATURE_GROUPS = {"ta_basic", "ta_advanced", "momentum", "volatility", "volume",
                         "price_position", "technical", "sentiment"}


def _validate_feature_groups(groups: list[str]) -> None:
    for g in groups:
        if g not in KNOWN_FEATURE_GROUPS:
            raise ValueError(f"unknown_feature_set:{g}")


def _train_target(req):
    from app.services.training_service import TrainingService
    return TrainingService().train(req)


def _train_direction_or_vol(ticker, label_type, feature_groups, horizon_days, model_type):
    from app.services.training_service import TrainRequest
    req = TrainRequest(
        tickers=[ticker],
        feature_groups=feature_groups,
        horizon_days=horizon_days,
        label_type=label_type,
        model_type=model_type,
        search_mode="none",  # NO Optuna — fair comparison
        save_model=True,
    )
    return _train_target(req)


def _train_meta(ticker, feature_groups, horizon_days, model_type):
    from app.services.meta_label_service import (
        MetaLabelTrainRequest, train_meta_label_model,
    )
    from app.services.primary_signal_service import PrimarySignalSpec
    req = MetaLabelTrainRequest(
        ticker=ticker,
        primary=PrimarySignalSpec(source="strategy", strategy_name="rsi_strategy"),
        tp_k=2.0, sl_k=1.0, timeout_days=horizon_days,
        vol_source="realized_sigma",
        cv_n_splits=5, cv_embargo_pct=0.01,
        model_type=model_type,
        lookback_days=730,
        feature_group=feature_groups,  # str | list[str] after P5 extension
    )
    return train_meta_label_model(req)


_PRIMARY_METRIC = {"direction": "auc", "volatility": "qlike", "meta_label": "auc_mean"}


def _extract_metrics(target: str, result) -> dict[str, float]:
    """Pull the relevant metrics from a training result into a flat dict."""
    if target == "meta_label":
        cv = result.get("cv_metrics", {}) if isinstance(result, dict) else {}
        return {
            "auc_mean": float(cv.get("auc_mean", 0.0)),
            "precision_at_50": float(cv.get("precision_at_50", 0.0)),
        }
    metrics = getattr(result, "metrics", None) or {}
    if target == "direction":
        return {
            "auc": float(metrics.get("test_auc", 0.0)),
            "f1": float(metrics.get("test_f1", 0.0)),
        }
    if target == "volatility":
        return {
            "qlike": float(metrics.get("test_qlike", 0.0)),
            "r2": float(metrics.get("test_r2", 0.0)),
            "mae": float(metrics.get("test_mae", 0.0)),
        }
    return {}


def _model_id(target: str, result) -> str:
    if target == "meta_label":
        return result.get("model_id", "?") if isinstance(result, dict) else "?"
    return getattr(result, "model_id", "?")


def run_ablation(
    *,
    ticker: str,
    targets: list[str],
    feature_sets: list[dict[str, Any]],
    horizon_days: int = 5,
    model_type: str = "xgboost",
) -> dict[str, Any]:
    if not targets or not feature_sets:
        raise ValueError("targets and feature_sets must each have ≥1 element")

    for fs in feature_sets:
        _validate_feature_groups(fs["groups"])

    t0 = time.time()
    matrix: dict[str, dict[str, dict[str, Any]]] = {t: {} for t in targets}

    for target in targets:
        for fs in feature_sets:
            try:
                if target == "meta_label":
                    res = _train_meta(ticker, fs["groups"], horizon_days, model_type)
                else:
                    res = _train_direction_or_vol(
                        ticker, target, fs["groups"], horizon_days, model_type
                    )
                metrics = _extract_metrics(target, res)
                cell = {"model_id": _model_id(target, res), **metrics}
                matrix[target][fs["name"]] = cell
            except Exception as e:
                logger.warning("ablation cell %s × %s failed: %s", target, fs["name"], e)
                matrix[target][fs["name"]] = {"error": str(e), "model_id": None}

    # Compute deltas relative to feature_sets[0]
    baseline_name = feature_sets[0]["name"]
    for target in targets:
        primary_metric = _PRIMARY_METRIC[target]
        baseline = matrix[target].get(baseline_name, {})
        baseline_val = baseline.get(primary_metric)
        if baseline_val is None:
            continue
        for fs in feature_sets[1:]:
            cell = matrix[target].get(fs["name"], {})
            if "error" in cell or primary_metric not in cell:
                continue
            cell[f"delta_{primary_metric}"] = cell[primary_metric] - baseline_val

    summary = _build_summary(matrix, targets, feature_sets)

    return {
        "ticker": ticker,
        "matrix": matrix,
        "summary": summary,
        "feature_sets_used": feature_sets,
        "model_type": model_type,
        "horizon_days": horizon_days,
        "elapsed_seconds": round(time.time() - t0, 2),
    }


def _build_summary(matrix, targets, feature_sets) -> dict[str, Any]:
    if len(feature_sets) < 2:
        return {"sentiment_helps_most": None, "interpretation": "Need ≥2 feature sets to compare."}
    fs1 = feature_sets[1]["name"]
    lifts = {}
    for target in targets:
        primary = _PRIMARY_METRIC[target]
        cell = matrix[target].get(fs1, {})
        delta_key = f"delta_{primary}"
        if delta_key in cell:
            # For QLIKE, lower is better → flip sign for "lift"
            sign = -1 if primary == "qlike" else 1
            lifts[target] = sign * cell[delta_key]
    if not lifts:
        return {"sentiment_helps_most": None, "interpretation": "No deltas computed."}
    best_target = max(lifts, key=lifts.get)
    return {
        "sentiment_helps_most": best_target,
        "interpretation": (
            f"Sentiment lifts {best_target}'s primary metric most "
            f"(deltas: {', '.join(f'{t}={v:+.3f}' for t, v in lifts.items())})."
        ),
    }
