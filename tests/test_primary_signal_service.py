"""Tests for PrimarySignalService dispatch (V4 Phase 3)."""
from __future__ import annotations

import pandas as pd
import pytest

from app.services.primary_signal_service import (
    PrimarySignalSpec,
    generate_primary_signals,
)


def _ohlc(n: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n)
    # Rising then falling pattern to trigger RSI
    half = n // 2
    closes = [100 + i * 0.5 for i in range(half)] + [
        100 + half * 0.5 - i * 0.8 for i in range(n - half)
    ]
    return pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": [c * 1.005 for c in closes],
        "low": [c * 0.995 for c in closes],
        "close": closes,
        "volume": [1_000_000] * n,
    })


def test_dispatch_strategy_name_returns_signal_series():
    ohlc = _ohlc(60)
    spec = PrimarySignalSpec(source="strategy", strategy_name="rsi_strategy")
    signals, strengths = generate_primary_signals(spec, ohlc)
    assert len(signals) == len(ohlc)
    assert set(signals.unique()).issubset({-1, 0, 1})
    # signal_strength is 1.0 for rule strategies at every position
    assert (strengths == 1.0).all()


def test_dispatch_model_id_loads_and_predicts(monkeypatch):
    ohlc = _ohlc(60)

    class FakeModel:
        def predict_proba(self, X):
            import numpy as np
            # Return proba[:, 1] alternating near 0.8 and 0.3
            probs = np.tile([0.3, 0.8], len(X) // 2 + 1)[: len(X)]
            return np.column_stack([1 - probs, probs])

    def fake_load_model(model_id: str):
        return FakeModel(), {"label_type": "direction", "task": "classification"}

    monkeypatch.setattr(
        "app.services.primary_signal_service._load_model_for_inference",
        fake_load_model,
    )
    spec = PrimarySignalSpec(source="model", primary_model_id="fake_model_id")
    signals, strengths = generate_primary_signals(spec, ohlc)
    assert len(signals) == len(ohlc)
    # sign of (proba - 0.5): 0.3 -> -1, 0.8 -> +1, alternating
    assert signals.iloc[1] == 1
    # strength = |proba - 0.5| * 2 -> 0.3->0.4, 0.8->0.6
    assert abs(strengths.iloc[1] - 0.6) < 1e-6


def test_both_strategy_and_model_raises():
    with pytest.raises(ValueError, match="exactly one"):
        PrimarySignalSpec(
            source="strategy",
            strategy_name="rsi_strategy",
            primary_model_id="some_id",
        )


def test_neither_strategy_nor_model_raises():
    with pytest.raises(ValueError, match="exactly one"):
        PrimarySignalSpec(source="strategy")
