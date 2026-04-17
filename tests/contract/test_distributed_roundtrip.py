from __future__ import annotations

"""
Contract test: end-to-end path from /predict to consumer /stats.

Uses direct in-process event injection (bypassing Kafka) since CI doesn't have
a Kafka broker. Verifies the data shape and logic of the consumer.
"""

from fastapi.testclient import TestClient


def test_events_consumer_stats_endpoint_contract():
    from app.workers.events_consumer import app, _stats
    from app.services.prediction_event_publisher import PredictionEvent

    _stats.clear()

    # Inject 3 bullish + 2 bearish events
    for pred, conf in [(1, 0.9), (1, 0.8), (0, 0.4), (1, 0.7), (0, 0.3)]:
        _stats["MSFT"].append(PredictionEvent(
            ticker="MSFT",
            prediction=pred,
            confidence=conf,
            model_id="model-x",
            model_type="ensemble",
        ))

    client = TestClient(app)
    resp = client.get("/stats/MSFT")
    assert resp.status_code == 200

    body = resp.json()
    # Contract: these fields MUST exist
    assert body["ticker"] == "MSFT"
    assert body["count"] == 5
    assert "avg_confidence" in body
    assert "bullish_ratio" in body
    assert "last_prediction_ts" in body

    # Values are correct
    assert body["bullish_ratio"] == 0.6  # 3/5
    assert abs(body["avg_confidence"] - 0.62) < 1e-9  # (0.9+0.8+0.4+0.7+0.3)/5


def test_consumer_case_insensitive_ticker():
    from app.workers.events_consumer import app, _stats
    from app.services.prediction_event_publisher import PredictionEvent

    _stats.clear()
    _stats["GOOG"].append(PredictionEvent(
        ticker="GOOG", prediction=1, confidence=0.7,
        model_id="m", model_type="logistic",
    ))

    client = TestClient(app)
    # lowercase query
    resp = client.get("/stats/goog")
    assert resp.status_code == 200
    assert resp.json()["count"] == 1
