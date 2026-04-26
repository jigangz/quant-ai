"""Tests for prediction_log repo (V4 P5 G1)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest
import os

from app.db.prediction_log import (
    PredictionLogRecord,
    LocalPredictionLogRepo,
    get_prediction_log_repo,
)


def _make_record(model_id="m1", ticker="MSFT", label_type="direction", **overrides):
    base = dict(
        model_id=model_id,
        ticker=ticker,
        label_type=label_type,
        horizon_days=5,
        predicted_value=0.7,
        predicted_signal=1,
        predicted_extras={"feature_group": "ta_basic"},
        resolve_at=datetime.now(timezone.utc) + timedelta(days=5),
    )
    base.update(overrides)
    return PredictionLogRecord(**base)


def test_local_repo_insert_and_get(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_LOCAL_PATH", str(tmp_path))
    repo = LocalPredictionLogRepo(storage_dir=tmp_path / "registry")
    rec = _make_record()
    repo.insert(rec)
    rows = repo.list_by_model_id("m1")
    assert len(rows) == 1
    assert rows[0].model_id == "m1"


def test_local_repo_list_unresolved_filters(tmp_path):
    repo = LocalPredictionLogRepo(storage_dir=tmp_path / "registry")
    past = datetime.now(timezone.utc) - timedelta(days=1)
    future = datetime.now(timezone.utc) + timedelta(days=10)
    r1 = _make_record(model_id="m1", resolve_at=past)
    r2 = _make_record(model_id="m1", resolve_at=future)
    r3 = _make_record(model_id="m1", resolve_at=past)
    repo.insert(r1); repo.insert(r2); repo.insert(r3)
    pending = repo.list_unresolved("m1", limit=10)
    assert len(pending) == 2  # only those with resolve_at < now and not resolved


def test_local_repo_update_resolution(tmp_path):
    repo = LocalPredictionLogRepo(storage_dir=tmp_path / "registry")
    rec = _make_record(resolve_at=datetime.now(timezone.utc) - timedelta(days=1))
    repo.insert(rec)
    repo.update_resolution(
        rec.id,
        actual_value=110.0,
        actual_return=0.05,
        is_correct=True,
        realized_R=0.6,
    )
    pending = repo.list_unresolved(rec.model_id, limit=10)
    assert len(pending) == 0
    rows = repo.list_by_model_id(rec.model_id)
    assert rows[0].is_correct is True
    assert rows[0].resolved_at is not None


def test_local_repo_list_by_model_id_window(tmp_path):
    repo = LocalPredictionLogRepo(storage_dir=tmp_path / "registry")
    old = datetime.now(timezone.utc) - timedelta(days=60)
    r_old = _make_record(model_id="m1")
    r_old_dict = r_old.model_dump()
    r_old_dict["created_at"] = old
    r_recent = _make_record(model_id="m1")
    repo.insert(PredictionLogRecord(**{**r_old_dict, "id": r_old.id}))
    repo.insert(r_recent)
    rows = repo.list_by_model_id("m1", since=datetime.now(timezone.utc) - timedelta(days=30))
    # Only recent is within 30d window
    assert len(rows) == 1


def test_factory_returns_local_when_no_supabase(tmp_path, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    repo = get_prediction_log_repo()
    assert type(repo).__name__ == "LocalPredictionLogRepo"
