"""Tests for triple-barrier labeling (V4 Phase 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.labels.meta_label import (
    TripleBarrierEvent,
    triple_barrier_events,
)


def _ohlc(dates, closes, highs=None, lows=None):
    highs = highs if highs is not None else [c * 1.01 for c in closes]
    lows = lows if lows is not None else [c * 0.99 for c in closes]
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [1_000_000] * len(closes),
    })


def test_tp_hit_gives_positive_r_and_correct_meta_label():
    # Price rises 10% on day 1 — TP (2 × 2% = 4%) hits long signal at day 0
    dates = pd.date_range("2026-01-01", periods=6)
    closes = [100.0, 110.0, 110.0, 110.0, 110.0, 110.0]
    ohlc = _ohlc(dates, closes, highs=[101.0, 111.0, 111.0, 111.0, 111.0, 111.0])
    signals = pd.Series([0, 0, 0, 0, 0, 0], index=ohlc.index)
    signals.iloc[0] = 1
    vol = pd.Series([0.02] * 6, index=ohlc.index)

    events = triple_barrier_events(
        ohlc, signals, vol, tp_k=2.0, sl_k=1.0, timeout_days=3
    )

    assert len(events) == 1
    ev = events.iloc[0]
    assert ev["t1_barrier"] == "tp"
    assert ev["realized_R"] == pytest.approx(2.0)
    assert ev["primary_direction_correct"] == 1


def test_sl_hit_gives_negative_r_and_wrong_meta_label():
    dates = pd.date_range("2026-01-01", periods=6)
    closes = [100.0, 97.0, 97.0, 97.0, 97.0, 97.0]
    ohlc = _ohlc(dates, closes, lows=[99.0, 96.5, 96.5, 96.5, 96.5, 96.5])
    signals = pd.Series([0] * 6, index=ohlc.index)
    signals.iloc[0] = 1  # long
    vol = pd.Series([0.02] * 6, index=ohlc.index)

    events = triple_barrier_events(ohlc, signals, vol, tp_k=2.0, sl_k=1.0, timeout_days=3)

    assert len(events) == 1
    ev = events.iloc[0]
    assert ev["t1_barrier"] == "sl"
    assert ev["realized_R"] == pytest.approx(-1.0)
    assert ev["primary_direction_correct"] == 0


def test_timeout_gives_fractional_r():
    # Price drifts +0.5% by day 3 — neither TP (4%) nor SL (2%) hits
    dates = pd.date_range("2026-01-01", periods=5)
    closes = [100.0, 100.2, 100.3, 100.5, 100.5]
    ohlc = _ohlc(dates, closes)
    signals = pd.Series([0] * 5, index=ohlc.index)
    signals.iloc[0] = 1
    vol = pd.Series([0.02] * 5, index=ohlc.index)

    events = triple_barrier_events(ohlc, signals, vol, tp_k=2.0, sl_k=1.0, timeout_days=3)

    assert len(events) == 1
    ev = events.iloc[0]
    assert ev["t1_barrier"] == "timeout"
    # realized_R = (close[t1] - close[t0]) / (sl_k * vol * close[t0])
    # = (100.5 - 100.0) / (1.0 * 0.02 * 100.0) = 0.005 / 0.02 = 0.25
    assert ev["realized_R"] == pytest.approx(0.25, abs=0.01)
    assert ev["primary_direction_correct"] == 1  # > 0 → correct


def test_zero_vol_degenerates_to_timeout():
    dates = pd.date_range("2026-01-01", periods=5)
    closes = [100.0, 100.5, 101.0, 100.7, 100.7]
    ohlc = _ohlc(dates, closes)
    signals = pd.Series([0] * 5, index=ohlc.index)
    signals.iloc[0] = 1
    vol = pd.Series([0.0] * 5, index=ohlc.index)  # zero vol → barriers at entry price

    events = triple_barrier_events(ohlc, signals, vol, tp_k=2.0, sl_k=1.0, timeout_days=3)

    # With zero vol, the barriers are both at entry_price; any price move triggers TP or SL.
    # Convention: zero-vol event is dropped (no meaningful barrier).
    assert len(events) == 0


def test_nan_ohlc_at_signal_time_is_skipped():
    dates = pd.date_range("2026-01-01", periods=5)
    closes = [100.0, np.nan, 105.0, 105.0, 105.0]
    ohlc = _ohlc(dates, closes)
    signals = pd.Series([0] * 5, index=ohlc.index)
    signals.iloc[1] = 1  # signal at NaN close
    vol = pd.Series([0.02] * 5, index=ohlc.index)

    events = triple_barrier_events(ohlc, signals, vol, tp_k=2.0, sl_k=1.0, timeout_days=3)
    assert len(events) == 0


def test_short_signal_symmetry_tp_on_drop():
    # Short signal: TP when price drops
    dates = pd.date_range("2026-01-01", periods=5)
    closes = [100.0, 95.0, 95.0, 95.0, 95.0]
    ohlc = _ohlc(dates, closes, lows=[99.0, 94.0, 94.0, 94.0, 94.0])
    signals = pd.Series([0] * 5, index=ohlc.index)
    signals.iloc[0] = -1  # short
    vol = pd.Series([0.02] * 5, index=ohlc.index)

    events = triple_barrier_events(ohlc, signals, vol, tp_k=2.0, sl_k=1.0, timeout_days=3)

    assert len(events) == 1
    ev = events.iloc[0]
    assert ev["t1_barrier"] == "tp"
    assert ev["realized_R"] == pytest.approx(2.0)
    assert ev["primary_direction_correct"] == 1


def test_primary_direction_correct_long_positive_return():
    # Direct target test: long + realized_R>0 → meta=1
    dates = pd.date_range("2026-01-01", periods=4)
    closes = [100.0, 104.1, 104.1, 104.1]
    ohlc = _ohlc(dates, closes, highs=[101.0, 105.0, 105.0, 105.0])
    signals = pd.Series([0] * 4, index=ohlc.index)
    signals.iloc[0] = 1
    vol = pd.Series([0.02] * 4, index=ohlc.index)
    events = triple_barrier_events(ohlc, signals, vol, tp_k=2.0, sl_k=1.0, timeout_days=2)
    assert events.iloc[0]["primary_direction_correct"] == 1


def test_primary_direction_correct_short_negative_return_is_wrong():
    # Short + price rose → realized_R<0 → meta=0
    dates = pd.date_range("2026-01-01", periods=4)
    closes = [100.0, 102.2, 102.2, 102.2]
    ohlc = _ohlc(dates, closes, highs=[103.0, 103.0, 103.0, 103.0])
    signals = pd.Series([0] * 4, index=ohlc.index)
    signals.iloc[0] = -1
    vol = pd.Series([0.02] * 4, index=ohlc.index)
    events = triple_barrier_events(ohlc, signals, vol, tp_k=2.0, sl_k=1.0, timeout_days=2)
    assert events.iloc[0]["realized_R"] < 0
    assert events.iloc[0]["primary_direction_correct"] == 0


def test_zero_signals_drop_before_barriering():
    dates = pd.date_range("2026-01-01", periods=5)
    closes = [100.0, 102.0, 104.0, 104.0, 104.0]
    ohlc = _ohlc(dates, closes)
    signals = pd.Series([0] * 5, index=ohlc.index)  # nothing triggers
    vol = pd.Series([0.02] * 5, index=ohlc.index)
    events = triple_barrier_events(ohlc, signals, vol, tp_k=2.0, sl_k=1.0, timeout_days=3)
    assert len(events) == 0
