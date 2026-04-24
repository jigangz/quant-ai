"""
Event feature builder for meta-labeling.

Given:
  - ohlc_ta: OHLC + existing TA features (e.g. from DatasetBuilder with ta_basic)
  - events: DataFrame with at least [event_time, primary_signal, signal_strength]
  - vol_series: pandas Series aligned to ohlc_ta, holding rolling or predicted vol
  - feature_cols: subset of ohlc_ta columns to use as base meta-model features

Produces a row per event with:
  - All feature_cols @ row-index(event_time) − 1   (lag 1 bar — no look-ahead)
  - signal_time_vol       = vol_series.iloc[row-index(event_time) − 1]
  - signal_strength       = events.signal_strength
  - time_since_last_signal = days since prior trigger of the same primary_source_key
                            (first event gets sentinel = len(ohlc_ta))

Returns DataFrame indexed 0..N-1 with columns [feature_cols..., signal_time_vol,
signal_strength, time_since_last_signal]. Callers build y separately.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def build_event_features(
    ohlc_ta: pd.DataFrame,
    events: pd.DataFrame,
    vol_series: pd.Series,
    primary_source_key: str,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Build per-event features for the meta-model."""
    if "event_time" not in events.columns:
        raise ValueError("events must have event_time column")
    if len(feature_cols) == 0:
        raise ValueError("feature_cols must not be empty")

    # Map event_time → row index in ohlc_ta
    ohlc_dates = pd.to_datetime(ohlc_ta["date"]).reset_index(drop=True)
    date_to_idx: dict[pd.Timestamp, int] = {
        pd.Timestamp(d): i for i, d in enumerate(ohlc_dates)
    }

    rows = []
    for ev in events.itertuples():
        t0 = pd.Timestamp(ev.event_time)
        if t0 not in date_to_idx:
            continue
        idx = date_to_idx[t0]
        lag_idx = max(0, idx - 1)
        row: dict[str, float] = {
            col: float(ohlc_ta[col].iloc[lag_idx]) for col in feature_cols
        }
        vol_at = vol_series.iloc[lag_idx] if len(vol_series) > lag_idx else 0.0
        row["signal_time_vol"] = (
            float(vol_at) if np.isfinite(vol_at) else 0.0
        )
        row["signal_strength"] = float(getattr(ev, "signal_strength", 1.0))
        rows.append({"__idx": idx, **row})

    if not rows:
        return pd.DataFrame(
            columns=list(feature_cols) + [
                "signal_time_vol", "signal_strength", "time_since_last_signal",
            ]
        )

    df = pd.DataFrame(rows).sort_values("__idx").reset_index(drop=True)

    # time_since_last_signal (days)
    deltas = []
    prev_idx: int | None = None
    sentinel = float(len(ohlc_ta))
    for cur_idx in df["__idx"]:
        if prev_idx is None:
            deltas.append(sentinel)
        else:
            deltas.append(float(cur_idx - prev_idx))
        prev_idx = int(cur_idx)
    df["time_since_last_signal"] = deltas
    df = df.drop(columns=["__idx"])

    # primary_source_key is metadata for upstream (different source keys → separate meta-model).
    _ = primary_source_key

    return df
