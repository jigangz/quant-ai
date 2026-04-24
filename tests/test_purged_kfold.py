"""Tests for Purged K-Fold CV splitter (V4 Phase 3 · López de Prado Ch.7)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.split.purged_kfold import PurgedKFold


def _events(n: int, span_days: int = 3) -> pd.DataFrame:
    """Build n events at consecutive daily timestamps with span_days holding time."""
    t0 = pd.date_range("2026-01-01", periods=n, freq="D")
    t1 = t0 + pd.Timedelta(days=span_days)
    return pd.DataFrame({"event_time": t0, "t1_hit_time": t1})


def test_n_splits_yields_expected_fold_count():
    events = _events(20)
    pkf = PurgedKFold(n_splits=5, embargo_pct=0.0)
    folds = list(pkf.split(events))
    assert len(folds) == 5
    # Each fold returns (train_idx, test_idx) numpy arrays
    for train_idx, test_idx in folds:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        assert len(set(train_idx) & set(test_idx)) == 0  # disjoint


def test_overlapping_events_are_purged_from_train():
    # 10 events, each spans 3 days. Test fold = indices 5,6.
    # Train indices whose [t0, t1] overlap with test [t0[5], t1[6]] must be purged.
    events = _events(10, span_days=3)
    pkf = PurgedKFold(n_splits=5, embargo_pct=0.0)
    folds = list(pkf.split(events))

    # Pick the fold with test indices [4, 5] (fold 2 of 5 for n=10)
    train_idx, test_idx = folds[2]
    test_t0 = events["event_time"].iloc[test_idx].min()
    test_t1 = events["t1_hit_time"].iloc[test_idx].max()

    for tr in train_idx:
        tr_t0 = events["event_time"].iloc[tr]
        tr_t1 = events["t1_hit_time"].iloc[tr]
        # No overlap: tr_t1 < test_t0 OR tr_t0 > test_t1 (plus embargo=0)
        assert tr_t1 < test_t0 or tr_t0 > test_t1, (
            f"event {tr} ({tr_t0}..{tr_t1}) overlaps test ({test_t0}..{test_t1})"
        )


def test_embargo_excludes_post_test_region_from_train():
    events = _events(20, span_days=1)
    # Fold 2 of 5 → test indices around [8, 11]
    pkf = PurgedKFold(n_splits=5, embargo_pct=0.10)  # 10% = 2 days
    folds = list(pkf.split(events))
    train_idx, test_idx = folds[2]

    test_t1_max = events["t1_hit_time"].iloc[test_idx].max()
    # embargo_pct=0.10 of 20 events × 1 day span = 2 days after test_t1_max are banned
    embargo_cutoff = test_t1_max + pd.Timedelta(days=2)

    for tr in train_idx:
        tr_t0 = events["event_time"].iloc[tr]
        # Post-test train events must be beyond embargo_cutoff
        if tr_t0 > test_t1_max:
            assert tr_t0 > embargo_cutoff


def test_empty_events_raises():
    events = _events(0)
    pkf = PurgedKFold(n_splits=5, embargo_pct=0.01)
    with pytest.raises(ValueError, match="empty"):
        list(pkf.split(events))


def test_fewer_events_than_splits_skips_empty_folds():
    events = _events(3)
    pkf = PurgedKFold(n_splits=5, embargo_pct=0.01)
    folds = list(pkf.split(events))
    # With n_splits=5 but only 3 events, at least 3 folds have valid tests.
    # Empty-test folds are skipped silently.
    assert 1 <= len(folds) <= 3
