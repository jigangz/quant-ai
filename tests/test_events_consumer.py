from __future__ import annotations

"""Tests for events_consumer FastAPI app."""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.services.prediction_event_publisher import PredictionEvent
from app.workers.events_consumer import _stats, app

client = TestClient(app)


def _make_event(ticker: str, prediction: int = 1, confidence: float = 0.7) -> PredictionEvent:
    return PredictionEvent(
        ticker=ticker,
        prediction=prediction,
        confidence=confidence,
        model_id="test_model",
        model_type="xgboost",
        timestamp=datetime.utcnow(),
    )


@pytest.fixture(autouse=True)
def clear_stats():
    """Reset in-memory stats before each test."""
    _stats.clear()
    yield
    _stats.clear()


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_stats_empty_ticker_returns_zero_count():
    resp = client.get("/stats/AAPL")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "AAPL"
    assert data["count"] == 0


def test_stats_with_events_calculates_correctly():
    """Inject events directly into _stats and verify /stats response."""
    ticker = "AAPL"
    _stats[ticker].append(_make_event(ticker, prediction=1, confidence=0.8))
    _stats[ticker].append(_make_event(ticker, prediction=1, confidence=0.6))
    _stats[ticker].append(_make_event(ticker, prediction=0, confidence=0.4))

    resp = client.get(f"/stats/{ticker}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["count"] == 3
    assert abs(data["avg_confidence"] - (0.8 + 0.6 + 0.4) / 3) < 0.001
    assert abs(data["bullish_ratio"] - 2 / 3) < 0.001
    assert "last_prediction_ts" in data


def test_stats_case_insensitive_ticker():
    """Lowercase ticker query should match uppercase _stats key."""
    _stats["AAPL"].append(_make_event("AAPL", prediction=1, confidence=0.9))

    resp = client.get("/stats/aapl")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "AAPL"
    assert data["count"] == 1


def test_stats_bullish_ratio_all_bearish():
    """bullish_ratio is 0.0 when all events are bearish."""
    ticker = "TSLA"
    _stats[ticker].append(_make_event(ticker, prediction=0, confidence=0.6))
    _stats[ticker].append(_make_event(ticker, prediction=0, confidence=0.7))

    resp = client.get(f"/stats/{ticker}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["bullish_ratio"] == 0.0
    assert data["count"] == 2
