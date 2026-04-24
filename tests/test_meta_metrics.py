"""Tests for meta-label metrics (V4 Phase 3)."""
from __future__ import annotations

import numpy as np

from app.backtest.metrics import calculate_meta_label_metrics


def test_precision_at_k_and_hit_rate():
    # 4 events. y_true = [1,1,0,0]; proba = [0.9, 0.8, 0.7, 0.2]
    # "trade" when score >= 0.5 -> events 0,1,2 -> precision = 2/3 = 0.667
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.2])
    realized_r = np.array([2.0, 2.0, -1.0, -1.0])

    m = calculate_meta_label_metrics(
        y_true=y_true, y_proba=y_proba, realized_r=realized_r,
        trade_threshold=0.5,
    )

    assert abs(m["precision_at_threshold"] - 2 / 3) < 1e-6
    assert abs(m["hit_rate_when_trade"] - 2 / 3) < 1e-6
    # Expected R when trade: mean of realized_r for events where proba >= 0.5 = mean([2,2,-1]) = 1.0
    assert abs(m["expected_R_when_trade"] - 1.0) < 1e-6
    assert m["trade_count"] == 3


def test_zero_trades_gives_zero_metrics():
    y_true = np.array([1, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])  # all below 0.5
    realized_r = np.array([2.0, -1.0, -1.0, -1.0])
    m = calculate_meta_label_metrics(
        y_true=y_true, y_proba=y_proba, realized_r=realized_r,
        trade_threshold=0.5,
    )
    assert m["trade_count"] == 0
    assert m["precision_at_threshold"] == 0.0
    assert m["hit_rate_when_trade"] == 0.0
    assert m["expected_R_when_trade"] == 0.0


def test_auc_present_and_finite():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    y_proba = rng.random(100) * 0.5 + y_true * 0.3  # weakly informed
    realized_r = np.where(y_true == 1, 2.0, -1.0)
    m = calculate_meta_label_metrics(y_true, y_proba, realized_r, trade_threshold=0.5)
    assert "auc" in m
    assert 0.0 <= m["auc"] <= 1.0
