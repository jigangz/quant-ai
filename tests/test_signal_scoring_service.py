"""Tests for SignalScoringService (V4 Phase 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.signal_scoring_service import (
    SignalScoreRequest, score_signal,
)


@pytest.fixture
def _prepared(monkeypatch):
    """Wires up fakes for OHLC + meta-model + primary dispatch."""
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    ohlc = pd.DataFrame({
        "date": dates,
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
        "close": [100.0 + i * 0.01 for i in range(n)],
        "volume": [1_000_000] * n,
        "rsi_14": [50.0] * n, "macd": [0.0] * n,
    })

    class FakeModel:
        def predict_proba(self, X):
            # Always predict 0.71 for class 1
            return np.column_stack([np.full(len(X), 0.29), np.full(len(X), 0.71)])

    fake_cache_record = {
        "model": FakeModel(),
        "metadata": {"task": "classification", "label_type": "meta_label",
                     "feature_group": "ta_basic"},
        "extras": {"meta_label": {
            "primary": {"source": "strategy", "strategy_name": "rsi_strategy",
                        "strategy_params": {}, "primary_model_id": None},
            "barrier": {"tp_k": 2.0, "sl_k": 1.0,
                        "timeout_days": 5, "vol_source": "realized_sigma"},
            "cv": {"metrics": {"auc_mean": 0.61}},
            "feature_set": ["rsi_14", "macd", "signal_time_vol",
                            "signal_strength", "time_since_last_signal"],
        }},
    }

    monkeypatch.setattr(
        "app.services.signal_scoring_service._load_meta_model",
        lambda mid: fake_cache_record,
    )
    monkeypatch.setattr(
        "app.services.signal_scoring_service._fetch_ohlc_ta",
        lambda ticker, lookback, feature_group: ohlc.copy(),
    )
    return ohlc


def test_mode_a_explicit_signal(_prepared):
    req = SignalScoreRequest(
        ticker="AAPL", meta_model_id="meta_test",
        signal=1, timestamp="2024-10-01",
    )
    resp = score_signal(req)
    assert resp["triggered"] is True
    assert resp["signal"] == 1
    assert abs(resp["reliability_score"] - 0.71) < 1e-6
    assert resp["recommended_action"] == "trade"


def test_mode_b_auto_trigger_strategy_silent(_prepared, monkeypatch):
    def silent(spec, ohlc):
        return pd.Series(0, index=ohlc.index), pd.Series(1.0, index=ohlc.index)

    monkeypatch.setattr(
        "app.services.signal_scoring_service.generate_primary_signals", silent
    )
    req = SignalScoreRequest(
        ticker="AAPL", meta_model_id="meta_test",
        strategy_name="rsi_strategy",
    )
    resp = score_signal(req)
    assert resp["triggered"] is False
    assert resp["signal"] == 0


def test_mode_a_wins_when_signal_and_strategy_given(_prepared):
    req = SignalScoreRequest(
        ticker="AAPL", meta_model_id="meta_test",
        signal=1, timestamp="2024-10-01",
        strategy_name="rsi_strategy",
    )
    resp = score_signal(req)
    assert resp["triggered"] is True
    assert resp["signal"] == 1  # explicit signal wins
