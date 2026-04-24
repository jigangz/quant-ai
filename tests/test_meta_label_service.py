"""Tests for MetaLabelTrainingService orchestrator (V4 Phase 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.meta_label_service import (
    MetaLabelTrainRequest,
    train_meta_label_model,
)
from app.services.primary_signal_service import PrimarySignalSpec


class _FakeOHLC:
    """Minimal fixture that replaces yfinance in tests."""
    def __init__(self, ticker: str, lookback_days: int):
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        rng = np.random.default_rng(seed=sum(map(ord, ticker)))
        closes = 100 + np.cumsum(rng.normal(0, 1, n))
        self.df = pd.DataFrame({
            "date": dates,
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": [1_000_000] * n,
        })


@pytest.fixture
def fake_market_data(monkeypatch):
    def _fake_fetch(ticker: str, lookback_days: int = 730) -> pd.DataFrame:
        return _FakeOHLC(ticker, lookback_days).df
    monkeypatch.setattr(
        "app.services.meta_label_service._fetch_ohlc", _fake_fetch
    )


def test_train_end_to_end_with_rsi_strategy(fake_market_data, tmp_path, monkeypatch):
    # Avoid touching real ModelRegistry: patch registration to no-op
    monkeypatch.setattr(
        "app.services.meta_label_service._register_meta_model",
        lambda **kw: "meta_test_abc123",
    )
    req = MetaLabelTrainRequest(
        ticker="AAPL",
        primary=PrimarySignalSpec(source="strategy", strategy_name="rsi_strategy"),
        tp_k=2.0, sl_k=1.0, timeout_days=5,
        vol_source="realized_sigma",
        cv_n_splits=3, cv_embargo_pct=0.01,
        model_type="xgboost", search_mode="default",
        lookback_days=300, feature_group="ta_basic",
    )
    result = train_meta_label_model(req)

    assert result["success"] is True
    assert result["model_id"] == "meta_test_abc123"
    assert result["event_count"] >= 0
    assert "cv_metrics" in result


def test_insufficient_events_raises_400_equivalent(fake_market_data, monkeypatch):
    # Force primary to never trigger
    def never_trigger(spec, ohlc):
        return pd.Series(0, index=ohlc.index), pd.Series(1.0, index=ohlc.index)

    monkeypatch.setattr(
        "app.services.meta_label_service.generate_primary_signals", never_trigger
    )
    req = MetaLabelTrainRequest(
        ticker="AAPL",
        primary=PrimarySignalSpec(source="strategy", strategy_name="rsi_strategy"),
        tp_k=2.0, sl_k=1.0, timeout_days=5, vol_source="realized_sigma",
        cv_n_splits=3, cv_embargo_pct=0.01, model_type="xgboost",
        lookback_days=200, feature_group="ta_basic",
    )
    with pytest.raises(ValueError, match="insufficient_events"):
        train_meta_label_model(req)


def test_p1_vol_source_auto_fallback_when_no_vol_model(fake_market_data, monkeypatch):
    monkeypatch.setattr(
        "app.services.meta_label_service._load_p1_vol_series",
        lambda ticker, ohlc: None,  # simulate no vol model registered
    )
    monkeypatch.setattr(
        "app.services.meta_label_service._register_meta_model",
        lambda **kw: "meta_fallback_abc"
    )
    req = MetaLabelTrainRequest(
        ticker="AAPL",
        primary=PrimarySignalSpec(source="strategy", strategy_name="rsi_strategy"),
        tp_k=2.0, sl_k=1.0, timeout_days=5, vol_source="p1_model",
        cv_n_splits=3, cv_embargo_pct=0.01, model_type="xgboost",
        lookback_days=300, feature_group="ta_basic",
    )
    result = train_meta_label_model(req)
    warnings = result.get("warnings", [])
    assert any("fallback" in w.lower() and "realized" in w.lower() for w in warnings)
