"""Tests for event feature builder (V4 Phase 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.meta_label_features import build_event_features


def _ohlc_with_ta(n=30):
    dates = pd.date_range("2026-01-01", periods=n)
    closes = [100 + np.sin(i / 5) * 5 for i in range(n)]
    df = pd.DataFrame({
        "date": dates,
        "open": closes, "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes], "close": closes,
        "volume": [1_000_000] * n,
        # Pretend TA features already computed upstream
        "rsi_14": [50 + np.cos(i / 4) * 20 for i in range(n)],
        "macd": [0.5 + np.sin(i / 3) for i in range(n)],
    })
    return df


def test_features_sliced_at_event_time_minus_one():
    ohlc_ta = _ohlc_with_ta(20)
    events = pd.DataFrame({
        "event_time": [ohlc_ta["date"].iloc[5], ohlc_ta["date"].iloc[10]],
        "primary_signal": [1, -1],
        "signal_strength": [1.0, 1.0],
    })
    vol_series = pd.Series([0.02] * 20, index=ohlc_ta.index)

    X = build_event_features(
        ohlc_ta=ohlc_ta, events=events, vol_series=vol_series,
        primary_source_key="rsi_strategy",
        feature_cols=["rsi_14", "macd"],
    )

    assert len(X) == 2
    # Feature values should come from row index 4 (event at 5, lag 1) and row 9
    assert abs(X["rsi_14"].iloc[0] - ohlc_ta["rsi_14"].iloc[4]) < 1e-9
    assert abs(X["rsi_14"].iloc[1] - ohlc_ta["rsi_14"].iloc[9]) < 1e-9


def test_signal_time_vol_feature_from_lagged_series():
    ohlc_ta = _ohlc_with_ta(15)
    events = pd.DataFrame({
        "event_time": [ohlc_ta["date"].iloc[7]],
        "primary_signal": [1],
        "signal_strength": [0.8],
    })
    vol_series = pd.Series([0.0] * 15, index=ohlc_ta.index)
    vol_series.iloc[6] = 0.04  # lagged vol should be this

    X = build_event_features(
        ohlc_ta=ohlc_ta, events=events, vol_series=vol_series,
        primary_source_key="model:fake", feature_cols=["rsi_14", "macd"],
    )
    assert abs(X["signal_time_vol"].iloc[0] - 0.04) < 1e-9
    assert abs(X["signal_strength"].iloc[0] - 0.8) < 1e-9


def test_time_since_last_signal_per_source():
    ohlc_ta = _ohlc_with_ta(20)
    events = pd.DataFrame({
        "event_time": [ohlc_ta["date"].iloc[3], ohlc_ta["date"].iloc[8],
                       ohlc_ta["date"].iloc[15]],
        "primary_signal": [1, -1, 1],
        "signal_strength": [1.0, 1.0, 1.0],
    })
    vol_series = pd.Series([0.02] * 20, index=ohlc_ta.index)

    X = build_event_features(
        ohlc_ta=ohlc_ta, events=events, vol_series=vol_series,
        primary_source_key="rsi_strategy", feature_cols=["rsi_14"],
    )
    # First event → time_since = large sentinel (e.g. len(ohlc_ta))
    assert X["time_since_last_signal"].iloc[0] == pytest.approx(20)  # sentinel
    # Second event at day 8, prior at day 3 → 5 days
    assert X["time_since_last_signal"].iloc[1] == 5
    # Third event at day 15, prior at day 8 → 7 days
    assert X["time_since_last_signal"].iloc[2] == 7
