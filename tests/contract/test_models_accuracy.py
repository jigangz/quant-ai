"""Contract tests for GET /models/{id}/accuracy (V4 P5 G1)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from app.main import app
    from app.services import accuracy_service

    monkeypatch.setattr(accuracy_service, "resolve_pending",
                         lambda model_id, limit=100: {"checked": 0, "newly_resolved": 0, "errors": 0})
    monkeypatch.setattr(accuracy_service, "aggregate",
                         lambda model_id, window_days=30: {
                             "total_predictions": 47, "resolved": 35, "pending": 12,
                             "hit_rate": 0.571, "avg_realized_R": 0.18,
                             "best_R": 1.85, "worst_R": -0.94,
                             "mae": None, "rmse": None,
                         })
    monkeypatch.setattr(accuracy_service, "by_ticker",
                         lambda model_id, window_days=30: [
                             {"ticker": "MSFT", "total": 35, "resolved": 25,
                              "hit_rate": 0.571, "avg_R": 0.18}
                         ])
    monkeypatch.setattr(accuracy_service, "last_predictions",
                         lambda model_id, limit=20: [])
    monkeypatch.setattr(
        "app.api.accuracy._get_model_record",
        lambda mid: {"label_type": "meta_label"} if mid == "ok_model" else None,
    )
    return TestClient(app)


def test_200_with_data(client):
    resp = client.get("/models/ok_model/accuracy?window_days=30")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ok_model"
    assert body["label_type"] == "meta_label"
    assert body["stats"]["hit_rate"] == 0.571


def test_404_not_found(client):
    resp = client.get("/models/missing/accuracy")
    assert resp.status_code == 404


def test_resolve_param_default_true(client, monkeypatch):
    from app.services import accuracy_service
    called = {}

    def _resolve(model_id, limit=100):
        called["yes"] = True
        return {"checked": 1, "newly_resolved": 1, "errors": 0}

    monkeypatch.setattr(accuracy_service, "resolve_pending", _resolve)
    resp = client.get("/models/ok_model/accuracy")
    assert resp.status_code == 200
    assert called.get("yes") is True


def test_window_days_clamped(client):
    resp = client.get("/models/ok_model/accuracy?window_days=400")
    assert resp.status_code == 422  # > 365 max


def test_resolve_false_skips_resolution(client, monkeypatch):
    from app.services import accuracy_service
    called = {"yes": False}
    monkeypatch.setattr(accuracy_service, "resolve_pending",
                         lambda *a, **k: called.update(yes=True) or {"checked": 0, "newly_resolved": 0, "errors": 0})
    resp = client.get("/models/ok_model/accuracy?resolve=false")
    assert resp.status_code == 200
    assert called["yes"] is False
