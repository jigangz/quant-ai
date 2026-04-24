"""
Meta-labeling · triple-barrier label generator.

Reference: López de Prado, Advances in Financial Machine Learning, Ch.3.

The triple-barrier method labels each primary signal by which of three barriers
is hit first:
  - Upper TP (profit target, in the trade's favor)
  - Lower SL (stop loss, against the trade)
  - Timeout (time barrier, neither TP nor SL reached)

For each signal at time t0 with direction d ∈ {+1, −1}:
  vol_at_t0  = vol_series[t0 − 1]     (lagged 1 bar — no look-ahead)
  tp_price   = close[t0] × (1 + d × tp_k × vol_at_t0)
  sl_price   = close[t0] × (1 − d × sl_k × vol_at_t0)
  t1         = first bar in (t0, t0 + timeout_days] where high/low touches tp/sl
  realized_R = +tp_k at TP, −sl_k at SL, fractional at timeout
  primary_direction_correct = (realized_R > 0)   ← meta-label target

Ambiguity handling: if a bar's [low, high] spans BOTH tp and sl in the same bar,
assume SL hit first (conservative). Zero-vol events are dropped. NaN OHLC at
signal time also drops the event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


@dataclass
class TripleBarrierEvent:
    """One labeled event for the meta-model."""
    event_time: pd.Timestamp
    primary_signal: int
    signal_strength: float
    entry_price: float
    tp_price: float
    sl_price: float
    timeout_time: pd.Timestamp
    t1_hit_time: pd.Timestamp
    t1_barrier: Literal["tp", "sl", "timeout"]
    realized_R: float
    primary_direction_correct: int


def triple_barrier_events(
    ohlc: pd.DataFrame,
    signals: pd.Series,
    vol_series: pd.Series,
    tp_k: float,
    sl_k: float,
    timeout_days: int,
    signal_strengths: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Generate triple-barrier events for every non-zero signal.

    Args:
        ohlc: DataFrame with columns ["date", "open", "high", "low", "close"]
              indexed by row position (0..N-1). date is a pandas Timestamp column.
        signals: Series of same length as ohlc with values in {-1, 0, +1}.
        vol_series: Series of same length as ohlc. Rolling or predicted volatility
                    (e.g. 0.02 for 2%). Indexed identically to ohlc.
        tp_k: Take-profit multiplier (barrier distance = tp_k × vol × entry_price).
        sl_k: Stop-loss multiplier.
        timeout_days: Max holding days from signal → timeout.
        signal_strengths: Optional Series of same length as ohlc, values in [0, 1].
                          Used when primary is ML (|proba−0.5|×2). Defaults to 1.0
                          for rule signals.

    Returns:
        DataFrame where each row is a TripleBarrierEvent (columns match dataclass
        fields). Empty DataFrame if no valid events.
    """
    if len(ohlc) != len(signals) or len(ohlc) != len(vol_series):
        raise ValueError(
            f"length mismatch: ohlc={len(ohlc)}, signals={len(signals)}, "
            f"vol={len(vol_series)}"
        )

    if signal_strengths is None:
        signal_strengths = pd.Series(1.0, index=ohlc.index)

    events: list[dict] = []
    n = len(ohlc)
    closes = ohlc["close"].to_numpy()
    highs = ohlc["high"].to_numpy()
    lows = ohlc["low"].to_numpy()
    dates = pd.to_datetime(ohlc["date"]).to_numpy()
    sigs = signals.to_numpy()
    vols = vol_series.to_numpy()
    strs = signal_strengths.to_numpy()

    for i in range(n):
        d = int(sigs[i])
        if d == 0:
            continue

        entry = closes[i]
        if not np.isfinite(entry):
            continue  # NaN OHLC → drop event

        # Lagged vol to avoid look-ahead; if i==0 fall back to current-bar vol.
        vol_at = vols[i - 1] if i > 0 else vols[i]
        if not np.isfinite(vol_at) or vol_at <= 0:
            continue  # zero-vol / NaN vol → drop event

        tp_price = entry * (1 + d * tp_k * vol_at)
        sl_price = entry * (1 - d * sl_k * vol_at)
        timeout_idx = min(i + timeout_days, n - 1)

        # Walk forward bars (i+1 .. timeout_idx), check TP/SL hit
        t1_idx = None
        t1_barrier: Literal["tp", "sl", "timeout"] = "timeout"
        for j in range(i + 1, timeout_idx + 1):
            bar_high = highs[j]
            bar_low = lows[j]
            if not (np.isfinite(bar_high) and np.isfinite(bar_low)):
                continue
            hit_tp = (bar_high >= tp_price) if d == 1 else (bar_low <= tp_price)
            hit_sl = (bar_low <= sl_price) if d == 1 else (bar_high >= sl_price)
            if hit_tp and hit_sl:
                # Same-bar ambiguity: assume SL hit first (conservative)
                t1_idx = j
                t1_barrier = "sl"
                break
            if hit_tp:
                t1_idx = j
                t1_barrier = "tp"
                break
            if hit_sl:
                t1_idx = j
                t1_barrier = "sl"
                break

        if t1_idx is None:
            t1_idx = timeout_idx
            t1_barrier = "timeout"

        # realized_R in trade's favor frame
        if t1_barrier == "tp":
            realized_r = float(tp_k)
        elif t1_barrier == "sl":
            realized_r = float(-sl_k)
        else:  # timeout
            exit_close = closes[t1_idx]
            if not np.isfinite(exit_close):
                continue
            realized_r = float(
                d * (exit_close - entry) / (sl_k * vol_at * entry)
            )

        events.append({
            "event_time": pd.Timestamp(dates[i]),
            "primary_signal": d,
            "signal_strength": float(strs[i]),
            "entry_price": float(entry),
            "tp_price": float(tp_price),
            "sl_price": float(sl_price),
            "timeout_time": pd.Timestamp(dates[timeout_idx]),
            "t1_hit_time": pd.Timestamp(dates[t1_idx]),
            "t1_barrier": t1_barrier,
            "realized_R": realized_r,
            "primary_direction_correct": int(realized_r > 0),
        })

    if not events:
        return pd.DataFrame(columns=[
            "event_time", "primary_signal", "signal_strength",
            "entry_price", "tp_price", "sl_price",
            "timeout_time", "t1_hit_time", "t1_barrier",
            "realized_R", "primary_direction_correct",
        ])
    return pd.DataFrame(events)
