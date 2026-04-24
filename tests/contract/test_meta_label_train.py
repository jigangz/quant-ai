"""Contract tests for POST /api/meta-label/train (V4 Phase 3)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from app.main import app
    from app.services import meta_label_service

    def fake_train(req):
        return {
            "success": True, "model_id": "meta_aapl_abc123", "registered": True,
            "event_count": 120, "class_balance": {"correct": 60, "wrong": 60},
            "cv_metrics": {
                "auc_mean": 0.60, "auc_std": 0.04, "precision_at_50": 0.65,
                "expected_R_when_trade": 0.4, "hit_rate_when_trade": 0.55,
                "folds_used": 5,
            },
            "barrier_config_used": {
                "tp_k": 2.0, "sl_k": 1.0, "timeout_days": 5, "vol_source": "realized_sigma"
            },
            "warnings": [],
        }
    monkeypatch.setattr(meta_label_service, "train_meta_label_model", fake_train)
    return TestClient(app)


def test_200_strategy_primary(client):
    resp = client.post("/api/meta-label/train", json={
        "ticker": "AAPL",
        "primary": {"source": "strategy", "strategy_name": "rsi_strategy"},
        "barrier": {"tp_k": 2.0, "sl_k": 1.0, "timeout_days": 5, "vol_source": "realized_sigma"},
        "cv": {"n_splits": 5, "embargo_pct": 0.01},
        "model": {"type": "xgboost"},
        "window": {"lookback_days": 730, "feature_group": "ta_basic"},
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["model_id"] == "meta_aapl_abc123"
    assert body["event_count"] == 120


def test_200_model_primary(client):
    resp = client.post("/api/meta-label/train", json={
        "ticker": "AAPL",
        "primary": {"source": "model", "primary_model_id": "dir_model_abc"},
        "barrier": {"tp_k": 2.0, "sl_k": 1.0, "timeout_days": 5, "vol_source": "realized_sigma"},
        "cv": {"n_splits": 5, "embargo_pct": 0.01},
        "model": {"type": "xgboost"},
        "window": {"lookback_days": 730, "feature_group": "ta_basic"},
    })
    assert resp.status_code == 200


def test_422_invalid_tp_k(client):
    resp = client.post("/api/meta-label/train", json={
        "ticker": "AAPL",
        "primary": {"source": "strategy", "strategy_name": "rsi_strategy"},
        "barrier": {"tp_k": -1.0, "sl_k": 1.0, "timeout_days": 5, "vol_source": "realized_sigma"},
        "cv": {"n_splits": 5, "embargo_pct": 0.01},
        "model": {"type": "xgboost"},
        "window": {"lookback_days": 730, "feature_group": "ta_basic"},
    })
    assert resp.status_code == 422


def test_400_insufficient_events_reraises(client, monkeypatch):
    from app.services import meta_label_service

    def fail(req):
        raise ValueError("insufficient_events: only 12 events")

    monkeypatch.setattr(meta_label_service, "train_meta_label_model", fail)
    resp = client.post("/api/meta-label/train", json={
        "ticker": "AAPL",
        "primary": {"source": "strategy", "strategy_name": "rsi_strategy"},
        "barrier": {"tp_k": 2.0, "sl_k": 1.0, "timeout_days": 5, "vol_source": "realized_sigma"},
        "cv": {"n_splits": 5, "embargo_pct": 0.01},
        "model": {"type": "xgboost"},
        "window": {"lookback_days": 730, "feature_group": "ta_basic"},
    })
    assert resp.status_code == 400
    assert "insufficient_events" in resp.json()["detail"]


def test_400_primary_source_conflict(client):
    resp = client.post("/api/meta-label/train", json={
        "ticker": "AAPL",
        "primary": {
            "source": "strategy",
            "strategy_name": "rsi_strategy",
            "primary_model_id": "some_model",  # conflict
        },
        "barrier": {"tp_k": 2.0, "sl_k": 1.0, "timeout_days": 5, "vol_source": "realized_sigma"},
        "cv": {"n_splits": 5, "embargo_pct": 0.01},
        "model": {"type": "xgboost"},
        "window": {"lookback_days": 730, "feature_group": "ta_basic"},
    })
    assert resp.status_code == 422  # pydantic validation catches via PrimarySignalSpec
