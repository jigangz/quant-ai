"""
Purged K-Fold cross-validation for event-indexed data.

Reference: López de Prado, Advances in Financial Machine Learning, Ch.7.

Standard K-Fold leaks when events in the training set overlap (in time) with
events in the test set. Purged K-Fold fixes this by:
  1. Splitting event indices into n_splits contiguous chunks (time-ordered)
  2. For each fold, using one chunk as test
  3. Purging from train any event whose [event_time, t1_hit_time] overlaps
     the test fold's time range
  4. Applying an embargo (gap after test) to prevent short-horizon leakage

Each event is assumed to have two timestamps: event_time (t0) and t1_hit_time (t1).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


class PurgedKFold:
    """Event-aware K-Fold splitter with purging + embargo.

    Args:
        n_splits: number of folds (≥2).
        embargo_pct: fraction of total events to exclude after the test window
                     as an embargo zone (e.g. 0.01 = 1%).

    Iterating `split(events)` yields (train_idx, test_idx) tuples.
    `events` must have columns "event_time" and "t1_hit_time" (both Timestamps).
    Folds with an empty test set are skipped.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if embargo_pct < 0 or embargo_pct > 0.5:
            raise ValueError("embargo_pct must be in [0, 0.5]")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self, events: pd.DataFrame
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if events is None or len(events) == 0:
            raise ValueError("empty events DataFrame")
        if "event_time" not in events.columns or "t1_hit_time" not in events.columns:
            raise ValueError("events must have event_time + t1_hit_time columns")

        sorted_events = events.sort_values("event_time").reset_index(drop=True)
        n = len(sorted_events)
        positional_to_original = events.sort_values("event_time").index.to_numpy()
        t0 = sorted_events["event_time"]
        t1 = sorted_events["t1_hit_time"]

        # Embargo as a timedelta: embargo_pct × median event-to-event gap × n
        if n >= 2:
            gaps = (t0.shift(-1) - t0).dropna()
            median_gap = gaps.median() if len(gaps) else pd.Timedelta(days=1)
        else:
            median_gap = pd.Timedelta(days=1)
        embargo_delta = median_gap * max(1, int(round(self.embargo_pct * n)))

        # Contiguous chunks of positional indices
        chunk_bounds = np.linspace(0, n, self.n_splits + 1, dtype=int)
        for k in range(self.n_splits):
            test_lo, test_hi = chunk_bounds[k], chunk_bounds[k + 1]
            if test_hi <= test_lo:
                continue  # empty test fold
            test_pos = np.arange(test_lo, test_hi)
            if len(test_pos) == 0:
                continue

            test_t0_min = t0.iloc[test_pos].min()
            test_t1_max = t1.iloc[test_pos].max()
            embargo_until = test_t1_max + embargo_delta

            train_mask = np.ones(n, dtype=bool)
            train_mask[test_pos] = False

            # Purge overlapping events: any event whose [t0, t1] intersects
            # [test_t0_min, test_t1_max] must be dropped from train.
            for i in range(n):
                if not train_mask[i]:
                    continue
                ev_t0 = t0.iloc[i]
                ev_t1 = t1.iloc[i]
                if ev_t1 < test_t0_min:
                    # strictly before → keep
                    continue
                if ev_t0 > embargo_until:
                    # strictly after embargo → keep
                    continue
                # otherwise: overlaps OR within embargo → drop
                train_mask[i] = False

            train_pos = np.where(train_mask)[0]
            # Map back to ORIGINAL (pre-sort) indices for downstream usage
            train_idx = positional_to_original[train_pos]
            test_idx = positional_to_original[test_pos]
            yield train_idx, test_idx
