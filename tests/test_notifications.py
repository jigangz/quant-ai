"""
Tests for Notification System (Phase 2.5 Step 7.2 / 7.3)

Tests alert generation and notification delivery using InMemoryNotifier.
No real AWS, Discord, or email services needed.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock settings before any app imports
# ---------------------------------------------------------------------------
_mock_settings = MagicMock()
_mock_settings.QUEUE_BACKEND = "memory"
_mock_settings.NOTIFY_BACKEND = "memory"
_mock_settings.REDIS_URL = "redis://localhost:6379/0"
_mock_settings.SQS_ENDPOINT_URL = None
_mock_settings.SNS_ENDPOINT_URL = None
_mock_settings.AWS_REGION = "us-east-1"

_mock_settings_module = MagicMock()
_mock_settings_module.settings = _mock_settings
sys.modules.setdefault("app.core.settings", _mock_settings_module)


# ---------------------------------------------------------------------------
# Notifier factory tests
# ---------------------------------------------------------------------------


class TestNotifierFactory:
    """Test get_notifier() returns the correct backend."""

    def test_memory_backend(self):
        from app.infra.notify import InMemoryNotifier
        import app.infra.notify as notify_mod

        notify_mod._notifier_instance = None
        _mock_settings.NOTIFY_BACKEND = "memory"

        notifier = notify_mod.get_notifier()
        assert isinstance(notifier, InMemoryNotifier)

        notify_mod._notifier_instance = None

    def test_sns_backend(self):
        from app.infra.notify import SNSNotifier
        import app.infra.notify as notify_mod

        notify_mod._notifier_instance = None
        try:
            with patch.object(notify_mod, "settings", _mock_settings):
                _mock_settings.NOTIFY_BACKEND = "sns"
                notifier = notify_mod.get_notifier()
                assert isinstance(notifier, SNSNotifier)
        finally:
            notify_mod._notifier_instance = None
            _mock_settings.NOTIFY_BACKEND = "memory"

    def test_unknown_backend_raises(self):
        import app.infra.notify as notify_mod

        notify_mod._notifier_instance = None
        try:
            with patch.object(notify_mod, "settings", _mock_settings):
                _mock_settings.NOTIFY_BACKEND = "unknown"
                with pytest.raises(ValueError, match="Unknown notify backend"):
                    notify_mod.get_notifier()
        finally:
            notify_mod._notifier_instance = None
            _mock_settings.NOTIFY_BACKEND = "memory"


# ---------------------------------------------------------------------------
# InMemoryNotifier tests
# ---------------------------------------------------------------------------


class TestInMemoryNotifier:
    """Test InMemoryNotifier publish/subscribe."""

    @pytest.fixture
    def notifier(self):
        from app.infra.notify import InMemoryNotifier
        return InMemoryNotifier()

    @pytest.mark.asyncio
    async def test_publish_records_history(self, notifier):
        await notifier.publish("alerts", {"type": "test"}, subject="Test")
        assert len(notifier._history) == 1
        assert notifier._history[0].topic == "alerts"
        assert notifier._history[0].payload == {"type": "test"}

    @pytest.mark.asyncio
    async def test_subscribe_receives_notifications(self, notifier):
        received = []

        async def handler(notification):
            received.append(notification)

        await notifier.subscribe("alerts", handler)
        await notifier.publish("alerts", {"type": "price_breakout", "symbol": "AAPL"})

        assert len(received) == 1
        assert received[0].payload["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, notifier):
        received_a = []
        received_b = []

        async def handler_a(n):
            received_a.append(n)

        async def handler_b(n):
            received_b.append(n)

        await notifier.subscribe("alerts", handler_a)
        await notifier.subscribe("alerts", handler_b)
        await notifier.publish("alerts", {"type": "test"})

        assert len(received_a) == 1
        assert len(received_b) == 1

    @pytest.mark.asyncio
    async def test_topic_isolation(self, notifier):
        received = []

        async def handler(n):
            received.append(n)

        await notifier.subscribe("alerts", handler)
        await notifier.publish("training", {"event": "started"})

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_stop_clears_state(self, notifier):
        await notifier.publish("alerts", {"type": "test"})
        await notifier.stop()
        assert len(notifier._history) == 0
        assert len(notifier._handlers) == 0


# ---------------------------------------------------------------------------
# Alert notification tests
# ---------------------------------------------------------------------------


class TestAlertNotifications:
    """Test alert publishing functions."""

    @pytest.fixture(autouse=True)
    def setup_notifier(self):
        import app.infra.notify as notify_mod
        from app.infra.notify import InMemoryNotifier

        notify_mod._notifier_instance = None
        _mock_settings.NOTIFY_BACKEND = "memory"
        self.notifier = notify_mod.get_notifier()
        assert isinstance(self.notifier, InMemoryNotifier)
        yield
        notify_mod._notifier_instance = None

    @pytest.mark.asyncio
    async def test_price_breakout_alert(self):
        from app.notifications.alerts import publish_price_breakout

        await publish_price_breakout("AAPL", 185.50, 180.00, "above")

        assert len(self.notifier._history) == 1
        n = self.notifier._history[0]
        assert n.topic == "alerts"
        assert n.payload["type"] == "price_breakout"
        assert n.payload["symbol"] == "AAPL"
        assert n.payload["current_price"] == 185.50
        assert n.payload["threshold"] == 180.00
        assert "broke above" in n.payload["message"]

    @pytest.mark.asyncio
    async def test_rsi_overbought_alert(self):
        from app.notifications.alerts import publish_rsi_alert

        await publish_rsi_alert("TSLA", 75.5, "overbought")

        n = self.notifier._history[0]
        assert n.payload["type"] == "rsi_alert"
        assert n.payload["condition"] == "overbought"
        assert n.payload["rsi_value"] == 75.5

    @pytest.mark.asyncio
    async def test_rsi_oversold_alert(self):
        from app.notifications.alerts import publish_rsi_alert

        await publish_rsi_alert("NVDA", 22.3, "oversold")

        n = self.notifier._history[0]
        assert n.payload["condition"] == "oversold"

    @pytest.mark.asyncio
    async def test_macd_crossover_alert(self):
        from app.notifications.alerts import publish_macd_crossover

        await publish_macd_crossover("MSFT", 1.5, 1.2, "bullish")

        n = self.notifier._history[0]
        assert n.payload["type"] == "macd_crossover"
        assert n.payload["crossover_type"] == "bullish"
        assert "bullish crossover" in n.payload["message"]

    @pytest.mark.asyncio
    async def test_sentiment_anomaly_alert(self):
        from app.notifications.alerts import publish_sentiment_anomaly

        await publish_sentiment_anomaly("GME", 0.8, 0.2, 0.6)

        n = self.notifier._history[0]
        assert n.payload["type"] == "sentiment_anomaly"
        assert n.payload["direction"] == "positive"
        assert n.payload["shift_magnitude"] == 0.6


# ---------------------------------------------------------------------------
# Training event notification tests
# ---------------------------------------------------------------------------


class TestTrainingEventNotifications:
    """Test training lifecycle event publishing."""

    @pytest.fixture(autouse=True)
    def setup_notifier(self):
        import app.infra.notify as notify_mod

        notify_mod._notifier_instance = None
        _mock_settings.NOTIFY_BACKEND = "memory"
        self.notifier = notify_mod.get_notifier()
        yield
        notify_mod._notifier_instance = None

    @pytest.mark.asyncio
    async def test_training_started(self):
        from app.notifications.training_events import publish_training_started

        await publish_training_started("run-1", ["AAPL", "MSFT"], "xgboost")

        n = self.notifier._history[0]
        assert n.topic == "training"
        assert n.payload["event"] == "training_started"
        assert n.payload["tickers"] == ["AAPL", "MSFT"]

    @pytest.mark.asyncio
    async def test_training_completed(self):
        from app.notifications.training_events import publish_training_completed

        await publish_training_completed("model-1", "run-1", {"accuracy": 0.92})

        n = self.notifier._history[0]
        assert n.payload["event"] == "training_completed"
        assert n.payload["model_id"] == "model-1"
        assert n.payload["metrics"]["accuracy"] == 0.92

    @pytest.mark.asyncio
    async def test_training_failed(self):
        from app.notifications.training_events import publish_training_failed

        await publish_training_failed("run-2", "Out of memory")

        n = self.notifier._history[0]
        assert n.payload["event"] == "training_failed"
        assert n.payload["error"] == "Out of memory"

    @pytest.mark.asyncio
    async def test_model_promoted(self):
        from app.notifications.training_events import publish_model_promoted

        await publish_model_promoted("model-1", "my-xgboost", "admin")

        n = self.notifier._history[0]
        assert n.payload["event"] == "model_promoted"
        assert n.payload["promoted_by"] == "admin"


# ---------------------------------------------------------------------------
# Delivery handler tests
# ---------------------------------------------------------------------------


class TestDeliveryHandlers:
    """Test notification delivery handlers."""

    @pytest.fixture(autouse=True)
    def setup_notifier(self):
        import app.infra.notify as notify_mod

        notify_mod._notifier_instance = None
        _mock_settings.NOTIFY_BACKEND = "memory"
        self.notifier = notify_mod.get_notifier()
        yield
        notify_mod._notifier_instance = None

    @pytest.mark.asyncio
    async def test_register_delivery_handlers(self):
        from app.notifications.delivery import register_delivery_handlers

        await register_delivery_handlers()

        # Should have 3 handlers per topic (discord, email, websocket) × 2 topics
        assert len(self.notifier._handlers.get("alerts", [])) == 3
        assert len(self.notifier._handlers.get("training", [])) == 3

    @pytest.mark.asyncio
    async def test_email_delivery_logs(self, caplog):
        import logging
        from app.notifications.delivery import EmailDelivery
        from app.infra.notify import Notification

        delivery = EmailDelivery(recipient="test@example.com")
        notification = Notification(
            topic="alerts",
            payload={"type": "test", "message": "Test alert"},
            subject="Test Subject",
        )

        with caplog.at_level(logging.INFO):
            await delivery.handle(notification)

        assert "EMAIL STUB" in caplog.text
        assert "test@example.com" in caplog.text
        assert "Test Subject" in caplog.text

    @pytest.mark.asyncio
    async def test_discord_delivery_skips_without_url(self):
        from app.notifications.delivery import DiscordWebhookDelivery
        from app.infra.notify import Notification

        delivery = DiscordWebhookDelivery(webhook_url=None)
        notification = Notification(
            topic="alerts",
            payload={"message": "test"},
        )

        # Should not raise
        await delivery.handle(notification)

    @pytest.mark.asyncio
    async def test_discord_delivery_sends_webhook(self):
        from app.notifications.delivery import DiscordWebhookDelivery
        from app.infra.notify import Notification

        delivery = DiscordWebhookDelivery(webhook_url="https://discord.example.com/webhook")
        notification = Notification(
            topic="alerts",
            payload={"type": "price_breakout", "message": "AAPL broke above $180"},
            subject="Price Alert",
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await delivery.handle(notification)

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[0][0] == "https://discord.example.com/webhook"
        assert "embeds" in call_kwargs[1]["json"]

    @pytest.mark.asyncio
    async def test_websocket_delivery_broadcasts(self):
        from app.notifications.delivery import WebSocketDelivery
        from app.infra.notify import Notification

        delivery = WebSocketDelivery()

        ws1 = AsyncMock()
        ws2 = AsyncMock()
        delivery.add_connection(ws1)
        delivery.add_connection(ws2)

        notification = Notification(
            topic="training",
            payload={"event": "training_completed"},
        )

        await delivery.handle(notification)

        ws1.send_text.assert_called_once()
        ws2.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_delivery_removes_dead_connections(self):
        from app.notifications.delivery import WebSocketDelivery
        from app.infra.notify import Notification

        delivery = WebSocketDelivery()

        live_ws = AsyncMock()
        dead_ws = AsyncMock()
        dead_ws.send_text = AsyncMock(side_effect=ConnectionError("gone"))

        delivery.add_connection(live_ws)
        delivery.add_connection(dead_ws)

        notification = Notification(
            topic="alerts",
            payload={"type": "test"},
        )

        await delivery.handle(notification)

        assert dead_ws not in delivery._connections
        assert live_ws in delivery._connections

    @pytest.mark.asyncio
    async def test_discord_color_for_failure(self):
        from app.notifications.delivery import DiscordWebhookDelivery

        assert DiscordWebhookDelivery._color_for_topic("training", {"event": "training_failed"}) == 0xFF0000

    @pytest.mark.asyncio
    async def test_discord_color_for_success(self):
        from app.notifications.delivery import DiscordWebhookDelivery

        assert DiscordWebhookDelivery._color_for_topic("training", {"event": "training_completed"}) == 0x00FF00


# ---------------------------------------------------------------------------
# End-to-end: alert → delivery pipeline
# ---------------------------------------------------------------------------


class TestEndToEndNotificationPipeline:
    """Test full flow: alert published → delivery handlers receive it."""

    @pytest.fixture(autouse=True)
    def setup_notifier(self):
        import app.infra.notify as notify_mod

        notify_mod._notifier_instance = None
        _mock_settings.NOTIFY_BACKEND = "memory"
        self.notifier = notify_mod.get_notifier()
        yield
        notify_mod._notifier_instance = None

    @pytest.mark.asyncio
    async def test_alert_flows_to_delivery(self):
        from app.notifications.delivery import register_delivery_handlers, email_delivery
        from app.notifications.alerts import publish_price_breakout

        # Register handlers
        await register_delivery_handlers(email_recipient="test@example.com")

        # Publish an alert
        await publish_price_breakout("AAPL", 185.0, 180.0, "above")

        # The InMemoryNotifier calls handlers synchronously, so delivery
        # should have been invoked. Check the notification was recorded.
        assert len(self.notifier._history) == 1
        assert self.notifier._history[0].payload["type"] == "price_breakout"

    @pytest.mark.asyncio
    async def test_training_event_flows_to_delivery(self):
        from app.notifications.delivery import register_delivery_handlers
        from app.notifications.training_events import publish_training_completed

        await register_delivery_handlers()
        await publish_training_completed("model-1", "run-1", {"accuracy": 0.9})

        assert len(self.notifier._history) == 1
        assert self.notifier._history[0].payload["event"] == "training_completed"
