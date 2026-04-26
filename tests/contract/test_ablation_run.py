"""Contract tests for POST /api/ablation/run (V4 P5)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from app.main import app
    from app.services import ablation_service

    def _fake_run(*, ticker, targets, feature_sets, horizon_days, model_type):
        return {
            "ticker": ticker,
            "matrix": {t: {fs["name"]: {"model_id": f"{t}_x", "auc": 0.6}
                            for fs in feature_sets} for t in targets},
            "summary": {"sentiment_helps_most": "direction",
                        "interpretation": "ok"},
            "feature_sets_used": feature_sets,
            "model_type": model_type,
            "horizon_days": horizon_days,
            "elapsed_seconds": 1.2,
        }

    monkeypatch.setattr(ablation_service, "run_ablation", _fake_run)
    return TestClient(app)


def test_200_happy_path(client):
    resp = client.post("/api/ablation/run", json={
        "ticker": "MSFT",
        "targets": ["direction", "volatility", "meta_label"],
        "feature_sets": [
            {"name": "ta_basic", "groups": ["ta_basic"]},
            {"name": "ta_basic + sentiment", "groups": ["ta_basic", "sentiment"]},
        ],
        "horizon_days": 5,
        "model_type": "xgboost",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "MSFT"
    assert "matrix" in body
    assert "summary" in body
    assert "elapsed_seconds" in body


def test_422_invalid_horizon(client):
    resp = client.post("/api/ablation/run", json={
        "ticker": "MSFT",
        "targets": ["direction"],
        "feature_sets": [{"name": "ta_basic", "groups": ["ta_basic"]}],
        "horizon_days": 999,
        "model_type": "xgboost",
    })
    assert resp.status_code == 422


def test_400_unknown_feature_set(client, monkeypatch):
    from app.services import ablation_service

    def _raise(**kw):
        raise ValueError("unknown_feature_set:mystery")

    monkeypatch.setattr(ablation_service, "run_ablation", _raise)
    resp = client.post("/api/ablation/run", json={
        "ticker": "MSFT",
        "targets": ["direction"],
        "feature_sets": [
            {"name": "mystery", "groups": ["mystery"]},
            {"name": "mystery2", "groups": ["mystery2"]},
        ],
        "horizon_days": 5,
        "model_type": "xgboost",
    })
    assert resp.status_code == 400
    assert "unknown_feature_set" in resp.json()["detail"]


def test_response_includes_summary(client):
    resp = client.post("/api/ablation/run", json={
        "ticker": "MSFT",
        "targets": ["direction"],
        "feature_sets": [
            {"name": "a", "groups": ["ta_basic"]},
            {"name": "b", "groups": ["ta_basic", "sentiment"]},
        ],
        "horizon_days": 5,
        "model_type": "xgboost",
    })
    body = resp.json()
    assert "interpretation" in body["summary"]
