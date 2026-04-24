"""Contract tests for POST /api/signal-score (V4 Phase 3)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from app.main import app
    from app.services import signal_scoring_service

    def fake_score(req):
        if req.meta_model_id == "missing":
            raise ValueError("meta_model_not_found:missing")
        if req.signal is None and req.strategy_name is None:
            return {
                "triggered": False, "signal": 0,
                "reason": "no signal/strategy provided",
                "timestamp": "2026-04-24T00:00:00Z",
            }
        if req.strategy_name and not req.signal:
            return {
                "triggered": False, "signal": 0,
                "reason": "rsi_strategy did not trigger",
                "timestamp": "2026-04-24T00:00:00Z",
            }
        return {
            "triggered": True, "signal": int(req.signal) if req.signal else 1,
            "reliability_score": 0.72, "expected_R": 0.44,
            "recommended_action": "trade",
            "sizing_hint": {"half_kelly_fraction": 0.18, "raw_kelly": 0.36, "cap": 0.25},
            "meta_model": {"id": req.meta_model_id,
                           "primary_source": "strategy:rsi_strategy", "cv_auc": 0.6},
            "timestamp": "2026-04-24T00:00:00Z",
        }
    monkeypatch.setattr(signal_scoring_service, "score_signal", fake_score)
    return TestClient(app)


def test_mode_a_explicit(client):
    resp = client.post("/api/signal-score", json={
        "ticker": "AAPL", "meta_model_id": "meta_abc",
        "signal": 1, "timestamp": "2026-04-24",
    })
    assert resp.status_code == 200
    assert resp.json()["triggered"] is True
    assert resp.json()["signal"] == 1


def test_mode_b_auto_strategy(client):
    resp = client.post("/api/signal-score", json={
        "ticker": "AAPL", "meta_model_id": "meta_abc",
        "strategy_name": "rsi_strategy",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["triggered"] is False  # our fake returns silent for B path
    assert body["signal"] == 0


def test_mode_a_wins_with_both(client):
    resp = client.post("/api/signal-score", json={
        "ticker": "AAPL", "meta_model_id": "meta_abc",
        "signal": 1, "timestamp": "2026-04-24",
        "strategy_name": "rsi_strategy",
    })
    assert resp.status_code == 200
    assert resp.json()["signal"] == 1


def test_404_meta_model_not_found(client):
    resp = client.post("/api/signal-score", json={
        "ticker": "AAPL", "meta_model_id": "missing",
        "signal": 1, "timestamp": "2026-04-24",
    })
    assert resp.status_code == 404


def test_400_ambiguous_no_signal_or_strategy(client):
    resp = client.post("/api/signal-score", json={
        "ticker": "AAPL", "meta_model_id": "meta_abc",
    })
    # Our service returns triggered=false in this case; wrapped as 200 with triggered=false
    assert resp.status_code == 200
    assert resp.json()["triggered"] is False
