"""Tests for predict services writing prediction_log rows (V4 P5)."""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest


@pytest.fixture
def captured_inserts(monkeypatch):
    """Capture every PredictionLogRecord insert across services."""
    captured = []

    class _FakeRepo:
        def insert(self, record):
            captured.append(record)
            return record

    def _factory():
        return _FakeRepo()

    monkeypatch.setattr("app.db.prediction_log.get_prediction_log_repo", _factory)
    return captured


def test_predict_service_writes_log(captured_inserts, monkeypatch):
    from app.services import predict_service
    # Stub out underlying predict() machinery — we only care the log row gets written.
    monkeypatch.setattr(predict_service, "_run_legacy_predict", lambda **kw: {
        "ticker": "MSFT", "prediction": 1, "confidence": 0.71,
        "model_id": "dir_msft_a", "model_type": "xgboost",
        "horizon_days": 5,
    })
    # Direct call to the post-predict log-write helper:
    predict_service._write_prediction_log(
        ticker="MSFT", model_id="dir_msft_a", model_type="xgboost",
        label_type="direction", horizon_days=5,
        predicted_value=0.71, predicted_signal=1,
        feature_group="ta_basic",
    )
    assert len(captured_inserts) == 1
    rec = captured_inserts[0]
    assert rec.label_type == "direction"
    assert rec.predicted_signal == 1


def test_volatility_predict_writes_log(captured_inserts):
    from app.services import volatility_predict_service
    volatility_predict_service._write_prediction_log(
        ticker="MSFT", model_id="vol_msft_a", model_type="xgboost",
        horizon_days=5, predicted_value=0.18, feature_group="ta_basic",
    )
    assert len(captured_inserts) == 1
    rec = captured_inserts[0]
    assert rec.label_type == "volatility"
    assert rec.predicted_signal is None


def test_signal_scoring_mode_a_writes_log(captured_inserts):
    from app.services import signal_scoring_service
    signal_scoring_service._write_prediction_log(
        ticker="AAPL", model_id="meta_aapl_a", model_type="xgboost",
        horizon_days=5, predicted_value=0.71, predicted_signal=1,
        primary_source="strategy:rsi_strategy", expected_R=0.54,
        feature_group="ta_basic",
    )
    assert len(captured_inserts) == 1
    rec = captured_inserts[0]
    assert rec.label_type == "meta_label"
    assert rec.predicted_extras["expected_R"] == 0.54
    assert rec.predicted_extras["primary_source"] == "strategy:rsi_strategy"


def test_log_write_is_non_blocking_on_repo_failure(monkeypatch):
    from app.services import predict_service

    class _BrokenRepo:
        def insert(self, record):
            raise RuntimeError("supabase down")

    monkeypatch.setattr(
        "app.db.prediction_log.get_prediction_log_repo",
        lambda: _BrokenRepo(),
    )
    # Should swallow and NOT raise:
    predict_service._write_prediction_log(
        ticker="MSFT", model_id="dir", model_type="xgboost",
        label_type="direction", horizon_days=5,
        predicted_value=0.5, predicted_signal=1, feature_group="ta_basic",
    )
