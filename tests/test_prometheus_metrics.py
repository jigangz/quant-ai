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
