"""Contract tests for GET /api/meta-label/coverage (V4 Phase 4)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from app.main import app
    from app.services import signal_scoring_service

    FAKE_RECORDS = [
        {"model_id": "meta_msft_a", "extras": {"meta_label": {
            "primary": {"source": "strategy", "strategy_name": "rsi_strategy"},
            "cv": {"metrics": {"auc_mean": 0.619}},
            "event_count": 483,
        }}, "metadata": {"ticker": "MSFT", "label_type": "meta_label"}},
        {"model_id": "meta_googl_b", "extras": {"meta_label": {
            "primary": {"source": "strategy", "strategy_name": "rsi_strategy"},
            "cv": {"metrics": {"auc_mean": 0.607}},
            "event_count": 486,
        }}, "metadata": {"ticker": "GOOGL", "label_type": "meta_label"}},
        {"model_id": "meta_aapl_c", "extras": {"meta_label": {
            "primary": {"source": "strategy", "strategy_name": "rsi_strategy"},
            "cv": {"metrics": {"auc_mean": 0.420}},
            "event_count": 492,
        }}, "metadata": {"ticker": "AAPL", "label_type": "meta_label"}},
        {"model_id": "meta_msft_ma", "extras": {"meta_label": {
            "primary": {"source": "strategy", "strategy_name": "ma_cross"},
            "cv": {"metrics": {"auc_mean": 0.55}},
            "event_count": 200,
        }}, "metadata": {"ticker": "MSFT", "label_type": "meta_label"}},
    ]

    def fake_list_meta_records():
        return FAKE_RECORDS

    monkeypatch.setattr(
        signal_scoring_service, "_list_meta_records", fake_list_meta_records
    )
    return TestClient(app)


def test_200_rsi_strategy_three_models(client):
    resp = client.get("/api/meta-label/coverage?strategy=rsi_strategy")
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy_name"] == "rsi_strategy"
    assert body["count"] == 3
    assert body["max_auc"] == pytest.approx(0.619, abs=1e-3)
    assert abs(body["avg_auc"] - (0.619 + 0.607 + 0.420) / 3) < 1e-3
    assert set(body["tickers"]) == {"MSFT", "GOOGL", "AAPL"}
    assert len(body["models"]) == 3


def test_200_zero_coverage(client):
    resp = client.get("/api/meta-label/coverage?strategy=bollinger_breakout")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["max_auc"] is None
    assert body["avg_auc"] is None
    assert body["tickers"] == []
    assert body["models"] == []


def test_404_unknown_strategy(client):
    resp = client.get("/api/meta-label/coverage?strategy=not_real")
    assert resp.status_code == 404


def test_aggregation_math_correct(client):
    resp = client.get("/api/meta-label/coverage?strategy=ma_cross")
    body = resp.json()
    assert body["count"] == 1
    assert body["max_auc"] == pytest.approx(0.55, abs=1e-3)
    assert body["avg_auc"] == pytest.approx(0.55, abs=1e-3)


def test_malformed_record_is_skipped(client, monkeypatch):
    from app.services import signal_scoring_service

    def broken_records():
        return [
            {"model_id": "broken", "extras": {}, "metadata": {"ticker": "X", "label_type": "meta_label"}},
            {"model_id": "ok", "extras": {"meta_label": {
                "primary": {"source": "strategy", "strategy_name": "rsi_strategy"},
                "cv": {"metrics": {"auc_mean": 0.6}},
            }}, "metadata": {"ticker": "MSFT", "label_type": "meta_label"}},
        ]
    monkeypatch.setattr(
        signal_scoring_service, "_list_meta_records", broken_records
    )
    resp = client.get("/api/meta-label/coverage?strategy=rsi_strategy")
    assert resp.status_code == 200
    # The broken record is skipped; 1 valid model counted
    assert resp.json()["count"] == 1
