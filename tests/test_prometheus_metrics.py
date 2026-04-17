from __future__ import annotations

from prometheus_client import Counter, Histogram

from app.core.metrics import (
    MODEL_INFERENCE_SECONDS,
    PREDICT_CONFIDENCE,
    PREDICT_TOTAL,
)


def test_predict_total_is_counter():
    assert isinstance(PREDICT_TOTAL, Counter)


def test_predict_confidence_is_histogram():
    assert isinstance(PREDICT_CONFIDENCE, Histogram)


def test_model_inference_seconds_is_histogram():
    assert isinstance(MODEL_INFERENCE_SECONDS, Histogram)


def test_predict_total_increments():
    before = PREDICT_TOTAL.labels(ticker="AAPL", model_type="xgboost")._value.get()
    PREDICT_TOTAL.labels(ticker="AAPL", model_type="xgboost").inc()
    after = PREDICT_TOTAL.labels(ticker="AAPL", model_type="xgboost")._value.get()
    assert after == before + 1.0


def test_predict_confidence_observes():
    # Should not raise
    PREDICT_CONFIDENCE.labels(ticker="TSLA").observe(0.75)


def test_model_inference_seconds_context_manager():
    with MODEL_INFERENCE_SECONDS.labels(model_type="xgboost").time():
        pass  # simulates inference


def test_metrics_endpoint_returns_prometheus_format():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "# HELP" in body
    assert "# TYPE" in body


def test_metrics_endpoint_includes_custom_counter():
    from fastapi.testclient import TestClient
    from app.main import app

    PREDICT_TOTAL.labels(ticker="SPY", model_type="ensemble").inc()

    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "predict_total" in resp.text
    assert 'ticker="SPY"' in resp.text


def test_predict_service_increments_metrics_on_success():
    """Integration: PredictionService increments PREDICT_TOTAL after successful prediction."""
    from unittest.mock import MagicMock, patch

    import numpy as np

    from app.services.predict_service import PredictionService

    mock_model = MagicMock()
    mock_model.model_type = "logistic"
    mock_model.predict_proba.return_value = np.array([[0.4, 0.6]])
    mock_model.predict.return_value = np.array([1])

    mock_features = MagicMock()
    mock_features.__len__ = lambda self: 1

    before = PREDICT_TOTAL.labels(ticker="INTC", model_type="logistic")._value.get()

    with patch("app.services.predict_service.get_model", return_value=mock_model), patch.object(
        PredictionService, "_build_features", return_value=mock_features
    ):
        result = PredictionService().predict(ticker="INTC")

    after = PREDICT_TOTAL.labels(ticker="INTC", model_type="logistic")._value.get()
    assert after == before + 1.0
    assert result["success"] is True
    assert result["signal"] == "LONG"
