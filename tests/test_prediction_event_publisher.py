from __future__ import annotations

"""Tests for prediction_event_publisher."""

from unittest.mock import AsyncMock, patch

import pytest

from app.services.prediction_event_publisher import (
    PredictionEvent,
    _producer_state,
    publish_prediction_event,
    start_producer,
    stop_producer,
)


def _make_event(**kwargs) -> PredictionEvent:
    defaults = dict(
        ticker="AAPL",
        prediction=1,
        confidence=0.75,
        model_id="test_model",
        model_type="xgboost",
    )
    defaults.update(kwargs)
    return PredictionEvent(**defaults)


def test_prediction_event_model_valid():
    """PredictionEvent validates correctly."""
    event = _make_event()
    assert event.ticker == "AAPL"
    assert event.prediction == 1
    assert event.confidence == 0.75
    assert event.model_id == "test_model"
    assert event.model_type == "xgboost"
    assert event.timestamp is not None


@pytest.mark.asyncio
async def test_start_producer_noop_when_not_kafka():
    """start_producer is a no-op when BROKER_BACKEND != kafka."""
    _producer_state["started"] = False
    _producer_state["producer"] = None

    with patch("app.services.prediction_event_publisher.settings") as mock_settings:
        mock_settings.BROKER_BACKEND = "redis"
        await start_producer()

    assert _producer_state["producer"] is None
    assert _producer_state["started"] is False


@pytest.mark.asyncio
async def test_publish_prediction_event_noop_when_not_kafka():
    """publish_prediction_event is a silent no-op when BROKER_BACKEND != kafka."""
    event = _make_event()
    _producer_state["warned_no_kafka"] = False

    with patch("app.services.prediction_event_publisher.settings") as mock_settings:
        mock_settings.BROKER_BACKEND = "memory"
        # Should NOT raise
        await publish_prediction_event(event)


@pytest.mark.asyncio
async def test_publish_prediction_event_noop_when_no_producer():
    """publish_prediction_event is a no-op when kafka producer is not ready."""
    event = _make_event(ticker="TSLA")
    _producer_state["producer"] = None

    with patch("app.services.prediction_event_publisher.settings") as mock_settings:
        mock_settings.BROKER_BACKEND = "kafka"
        # Should NOT raise
        await publish_prediction_event(event)


@pytest.mark.asyncio
async def test_stop_producer_resets_state():
    """stop_producer stops the producer and resets state."""
    mock_producer = AsyncMock()
    _producer_state["producer"] = mock_producer
    _producer_state["started"] = True

    await stop_producer()

    assert _producer_state["producer"] is None
    assert _producer_state["started"] is False
    mock_producer.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_publish_handles_send_failure_silently():
    """publish_prediction_event does not raise when Kafka send fails."""
    mock_producer = AsyncMock()
    mock_producer.send_and_wait.side_effect = Exception("Kafka send error")
    _producer_state["producer"] = mock_producer

    event = _make_event(ticker="MSFT")

    with patch("app.services.prediction_event_publisher.settings") as mock_settings:
        mock_settings.BROKER_BACKEND = "kafka"
        # Should NOT raise despite send failure
        await publish_prediction_event(event)
