"""Tests for AccuracyService (V4 P5 G1)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest

from app.db.prediction_log import PredictionLogRecord


@pytest.fixture
def fake_ohlc(monkeypatch):
    """Make AccuracyService use a deterministic OHLC slice."""
    from app.services import accuracy_service

    def _fake_fetch(ticker, start, end):
        # Linear price 100 → 110 over 10 trading days
        idx = pd.date_range(start, end, freq="D")
        closes = np.linspace(100, 110, len(idx))
        return pd.DataFrame({"date": idx, "close": closes,
                              "open": closes, "high": closes * 1.01,
                              "low": closes * 0.99, "volume": [1_000_000] * len(idx)})

    monkeypatch.setattr(accuracy_service, "_fetch_ohlc_slice", _fake_fetch)


@pytest.fixture
def fake_repo(monkeypatch):
    from app.db import prediction_log

    rows = {}

    class _Fake:
        def insert(self, rec):
            rows[rec.id] = rec
            return rec

        def list_unresolved(self, model_id, limit=100):
            now = datetime.now(timezone.utc)
            return [r for r in rows.values()
                    if r.model_id == model_id and not r.resolved_at and r.resolve_at < now][:limit]

        def list_by_model_id(self, model_id, since=None, limit=500):
            return [r for r in rows.values() if r.model_id == model_id][:limit]

        def update_resolution(self, rid, **updates):
            r = rows.get(rid)
            if not r:
                return
            d = r.model_dump()
            d.update(updates)
            d["resolved_at"] = datetime.now(timezone.utc)
            rows[rid] = PredictionLogRecord(**d)

    fake = _Fake()
    monkeypatch.setattr(prediction_log, "get_prediction_log_repo", lambda: fake)
    return fake


def _make(model_id, label_type, ticker="MSFT", **k):
    base = dict(
        model_id=model_id, ticker=ticker, label_type=label_type,
        horizon_days=5, predicted_value=0.7,
        predicted_signal=(1 if label_type != "volatility" else None),
        resolve_at=datetime.now(timezone.utc) - timedelta(days=1),
        predicted_at=datetime.now(timezone.utc) - timedelta(days=6),
    )
    base.update(k)
    return PredictionLogRecord(**base)


def test_resolve_direction_correct_prediction(fake_repo, fake_ohlc):
    from app.services.accuracy_service import resolve_pending
    fake_repo.insert(_make("dir_a", "direction", predicted_signal=1))
    result = resolve_pending("dir_a", limit=10)
    assert result["newly_resolved"] == 1
    rows = fake_repo.list_by_model_id("dir_a")
    assert rows[0].is_correct is True  # price went up + predicted +1 = correct
    assert rows[0].realized_R is not None


def test_resolve_direction_wrong_prediction(fake_repo, fake_ohlc):
    from app.services.accuracy_service import resolve_pending
    fake_repo.insert(_make("dir_b", "direction", predicted_signal=-1))
    resolve_pending("dir_b", limit=10)
    rows = fake_repo.list_by_model_id("dir_b")
    assert rows[0].is_correct is False


def test_resolve_volatility_no_hit_miss(fake_repo, fake_ohlc):
    from app.services.accuracy_service import resolve_pending
    fake_repo.insert(_make("vol_a", "volatility", predicted_signal=None))
    resolve_pending("vol_a", limit=10)
    rows = fake_repo.list_by_model_id("vol_a")
    assert rows[0].is_correct is None  # vol target — no hit/miss
    assert rows[0].realized_R is None
    assert rows[0].actual_value is not None  # realized vol


def test_resolve_meta_label_uses_signal(fake_repo, fake_ohlc):
    from app.services.accuracy_service import resolve_pending
    fake_repo.insert(_make("meta_a", "meta_label", predicted_signal=1))
    resolve_pending("meta_a", limit=10)
    rows = fake_repo.list_by_model_id("meta_a")
    assert rows[0].is_correct is True


def test_aggregate_hit_rate(fake_repo, fake_ohlc):
    from app.services.accuracy_service import aggregate, resolve_pending
    for sig in [1, 1, -1, 1]:  # 3 of 4 will be "correct" since price rose
        fake_repo.insert(_make("agg_a", "direction", predicted_signal=sig))
    resolve_pending("agg_a", limit=10)
    stats = aggregate("agg_a", window_days=30)
    assert stats["resolved"] == 4
    assert abs(stats["hit_rate"] - 0.75) < 1e-6


def test_aggregate_avg_realized_R(fake_repo, fake_ohlc):
    from app.services.accuracy_service import aggregate, resolve_pending
    for _ in range(3):
        fake_repo.insert(_make("agg_b", "direction", predicted_signal=1))
    resolve_pending("agg_b", limit=10)
    stats = aggregate("agg_b", window_days=30)
    assert stats["avg_realized_R"] is not None
    assert stats["avg_realized_R"] > 0  # all "correct" + price up = positive R


def test_aggregate_pending_count(fake_repo):
    from app.services.accuracy_service import aggregate
    # Future resolve_at — never resolves
    fake_repo.insert(_make(
        "pend_a", "direction",
        resolve_at=datetime.now(timezone.utc) + timedelta(days=10),
    ))
    stats = aggregate("pend_a", window_days=30)
    assert stats["pending"] == 1
    assert stats["resolved"] == 0


def test_aggregate_empty_model(fake_repo):
    from app.services.accuracy_service import aggregate
    stats = aggregate("doesnt_exist", window_days=30)
    assert stats["total_predictions"] == 0
    assert stats["hit_rate"] is None
